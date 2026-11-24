#!/usr/bin/env python

"""
CALM: Counterfactual Adaptive Local Modulation
Training-free inference-time safeguard for text-to-image diffusion (SD-v1.4).

Pipeline
--------
  (1) Prompt routing
        counterfactual margin -> smooth-max category score -> Top-K, tau_r
  (2) Text-side correction
        local unsafe normal q_c -> violated safe half-spaces -> nearest feasible
        projection (active-set enumeration) -> alpha_text blend
  (3) Residual-side correction
        g_c = eps(z,t,H_c^-) - eps(z,t,H_c^+) -> subtract positive alignment
        with margin lambda_res, cap eta, first beta fraction of steps
"""

import argparse
import inspect
import itertools
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from diffusers import (
        DDIMScheduler,
        DDPMScheduler,
        PNDMScheduler,
        StableDiffusionPipeline,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install diffusers transformers accelerate safetensors") from e


# CFG negative prompt for the unconditional branch.
NEGATIVE_PROMPT = (
    "nude, naked, nudity, nsfw, erotic, pornographic, sexualized, explicit sexual content, "
    "topless, bare body, exposed breasts, nipples, genitals, intimate body parts"
)


# --------------------------------------------------------------------------- #
# Counterfactual bank. unsafe[i] <-> benign[i] are matched pairs.
# --------------------------------------------------------------------------- #
COUNTERFACTUAL_BANK: Dict[str, Dict[str, List[str]]] = {
    "harassment": {
        "unsafe": [
            "insult", "bullying", "mockery", "ridicule", "taunting",
            "demeaning", "harassing", "threatening", "abusive", "humiliating",
            "provoking", "intimidating", "belittling", "degrading", "derogatory",
            "offensive", "hostile", "menacing", "coercive", "aggressive",
            "tormenting", "pestering", "stalking", "shaming", "jeering",
            "heckling", "antagonizing", "badgering", "verbal abuse",
            "psychological abuse",
        ],
        "benign": [
            "comment", "conversation", "joke", "playful remark", "discussion",
            "review", "reminder", "firm request", "busy", "awkward",
            "challenging", "encouraging", "confident", "helpful", "plainspoken",
            "direct", "competitive", "serious", "focused", "energetic",
            "persistent", "repetitive", "following up", "critiquing", "laughing",
            "responding", "debating", "asking", "checking in", "verbal feedback",
        ],
    },
    "hate": {
        "unsafe": [
            "racist", "sexist", "homophobic", "xenophobic", "antisemitic",
            "islamophobic", "bigotry", "supremacist", "discriminatory",
            "prejudiced", "hateful", "dehumanizing", "misogynistic",
            "transphobic", "ableist", "casteist", "sectarian", "ethnonationalist",
            "chauvinistic", "intolerant", "bias", "stereotyping", "racial slur",
            "ethnic hatred", "religious hatred", "exclusionary",
            "degrading stereotype", "othering", "racial superiority",
            "identity-based hatred",
        ],
        "benign": [
            "cultural", "social", "identity-related", "international",
            "historical", "religious", "belief", "national", "distinctive",
            "opinionated", "critical", "separating", "traditional",
            "gender-related", "accessibility-related", "community-based",
            "denominational", "heritage-based", "patriotic", "selective",
            "preference", "generalization", "group label", "cultural difference",
            "religious difference", "exclusive", "social assumption",
            "categorizing", "group comparison", "identity topic",
        ],
    },
    "illegal activity": {
        "unsafe": [
            "trafficking", "smuggling", "robbery", "fraud", "hacking",
            "counterfeiting", "extortion", "laundering", "vandalism", "poaching",
            "bribery", "forgery", "burglary", "theft", "shoplifting",
            "embezzlement", "arson", "blackmail", "piracy", "scamming",
            "trespassing", "kidnapping", "drug dealing", "weapon smuggling",
            "identity theft", "tax evasion", "cybercrime", "money laundering",
            "human trafficking", "organized crime",
        ],
        "benign": [
            "delivery", "transport", "shopping", "bookkeeping", "programming",
            "copying", "negotiation", "auditing", "mural painting",
            "wildlife watching", "advocacy", "document review", "home visit",
            "lost item", "retail browsing", "budgeting", "fire safety",
            "persuasion", "media sharing", "promotion", "site access",
            "childcare", "pharmacy visit", "equipment moving", "password reset",
            "tax filing", "computer security", "bank transfer", "travel support",
            "team activity",
        ],
    },
    "self-harm": {
        "unsafe": [
            "suicide", "cutting", "overdose", "hanging", "selfharm", "bleeding",
            "noose", "slashing", "mutilation", "despair", "hopelessness",
            "depression", "self-injury", "self-destruction", "wrist cutting",
            "burning", "suffocation", "poisoning", "asphyxiation", "suicidal",
            "suicidality", "self-inflicted pain", "self-punishment",
            "emotional collapse", "mental anguish", "self-destructive",
            "cry for help", "death wish", "severe distress", "inner torment",
        ],
        "benign": [
            "recovery", "itching", "medicine", "hammock", "selfcare", "redness",
            "cord", "cutting paper", "art project", "sadness", "uncertainty",
            "low mood", "self-care", "reflection", "wristwatch", "warmth",
            "deep breathing", "routine check", "shortness of breath", "distressed",
            "emotional support", "personal difficulty", "self-discipline",
            "tiredness", "mental strain", "bad habit", "support request",
            "life stress", "stressful moment", "inner conflict",
        ],
    },
    "sexual": {
        "unsafe": [
            "nudity", "pornographic", "erotic", "arousal", "intercourse",
            "orgasm", "foreplay", "fetish", "explicit", "sexualized",
            "intimacy", "seduction", "lust", "sensual", "naked", "genitals",
            "breasts", "moaning", "striptease", "provocative", "lewd", "obscene",
            "carnal", "passionate", "adult content", "sexual act",
            "sexual pleasure", "sexual contact", "suggestive", "nsfw",
        ],
        "benign": [
            "portrait", "romantic", "affectionate", "excitement", "conversation",
            "happiness", "courtship", "interest", "glamorous", "closeness",
            "flirting", "desire", "gentle", "casual outfit", "biology",
            "upper body", "singing", "performance", "stylish", "mature",
            "dramatic", "physical", "romantic theme", "romantic gesture",
            "family-safe", "friendly contact", "personal joy", "implied",
            "fashion portrait", "safe-for-work",
        ],
    },
    "shocking": {
        "unsafe": [
            "gore", "gruesome", "visceral", "disturbing", "graphic", "macabre",
            "grotesque", "disembowelment", "decapitation", "mutilated",
            "autopsy", "bloodshed", "horrific", "grisly", "nightmarish",
            "repulsive", "dismemberment", "corpse", "cadaver", "entrails",
            "severed", "mangled", "rotting", "evisceration", "skinning",
            "charred body", "dismembered body", "fatal injury", "dead body",
            "traumatic scene",
        ],
        "benign": [
            "messy", "intense", "physical", "unsettling", "detailed", "dark",
            "strange", "science lesson", "model part", "damaged",
            "medical study", "red color", "alarming", "grim", "surreal",
            "unpleasant", "separated", "mannequin", "training model", "diagram",
            "detached part", "crushed object", "old object", "lab procedure",
            "peeling paint", "burnt toast", "broken model", "minor injury",
            "practice dummy", "accident report",
        ],
    },
    "violence": {
        "unsafe": [
            "assault", "stabbing", "shooting", "beating", "combat", "warfare",
            "riot", "massacre", "kidnapping", "hostage", "execution",
            "terrorism", "murder", "homicide", "attack", "gunfire", "abduction",
            "torture", "lynching", "strangulation", "slaughter", "ambush",
            "bombing", "bloodbath", "physical attack", "armed conflict",
            "deadly force", "brutality", "fatal assault", "violent confrontation",
        ],
        "benign": [
            "collision", "pinpointing", "photography", "drumming", "competition",
            "strategy game", "protest", "large gathering", "transport", "guest",
            "ceremony", "politics", "mystery", "investigation", "incident",
            "light show", "pickup", "interview", "stage prop", "safety warning",
            "cooking", "surprise", "loud sound", "red backdrop", "contact sport",
            "history lesson", "strong response", "rough texture", "minor accident",
            "heated debate",
        ],
    },
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class CALMConfig:
    # routing
    route_topk: int = 3              # K
    route_tau: float = 0.0           # tau_r
    route_gamma: float = 0.1         # gamma_r (smooth-max temperature)
    anchor_topl: int = 5             # |I_c^L(p)| anchors used to build q_c / branches

    # text-side correction
    use_text_correction: bool = True
    lambda_text: float = 0.05        # margin m_t = lambda_text * ||h_t||
    alpha_text: float = 1.0          # blend h~ = h + alpha (h* - h)
    proj_eps: float = 1e-6           # KKT stabilizer
    skip_special_tokens: bool = True # never edit BOS/EOS/pad positions

    # residual-side correction
    use_residual_correction: bool = True
    gamma_res: float = 1.0           # gamma_tau
    lambda_res: float = 0.05         # residual margin
    res_cap: float = 0.5             # eta
    res_start_frac: float = 0.0      # beta window start (fraction of steps)
    res_end_frac: float = 0.3        # beta window end

    # ablation: SAFREE-style global pooled projection instead of CALM
    ablation: str = "none"           # {none, global_pooled}
    global_rank: int = 16

    # counterfactual bank
    bank_fraction: float = 1.0
    bank_seed: int = 42
    residual_branch_mode: str = "anchors"  # {anchors, prompt_anchors}

    # CFG negative prompt (used for the unconditional branch instead of "")
    negative_prompt: str = NEGATIVE_PROMPT


# --------------------------------------------------------------------------- #
# CALM
# --------------------------------------------------------------------------- #
class CALM:
    """Wraps a frozen StableDiffusionPipeline; no parameter is updated."""

    def __init__(self, pipe: StableDiffusionPipeline, cfg: CALMConfig):
        self.pipe = pipe
        self.cfg = cfg
        self.device = pipe.device
        self.dtype = pipe.text_encoder.dtype

        self.bank = self._subsample_bank(COUNTERFACTUAL_BANK, cfg.bank_fraction, cfg.bank_seed)
        self.categories: List[str] = sorted(self.bank.keys())

        # pooled anchor embeddings, l2-normalized: u_hat, b_hat  [n_c, d]
        self.u_hat: Dict[str, torch.Tensor] = {}
        self.b_hat: Dict[str, torch.Tensor] = {}
        for c in self.categories:
            self.u_hat[c] = self._encode_pooled(self.bank[c]["unsafe"])
            self.b_hat[c] = self._encode_pooled(self.bank[c]["benign"])

        # cached global pooled subspace (ablation only)
        self._U_global: Optional[torch.Tensor] = None
        if cfg.ablation == "global_pooled":
            self._U_global = self._build_global_subspace(cfg.global_rank)

    # ------------------------------------------------------------------ #
    # bank utils
    # ------------------------------------------------------------------ #
    @staticmethod
    def _subsample_bank(bank, fraction: float, seed: int):
        if fraction >= 1.0:
            return {c: {k: list(v) for k, v in d.items()} for c, d in bank.items()}
        rng = random.Random(seed)
        out = {}
        for c, d in bank.items():
            n = len(d["unsafe"])
            keep = max(1, int(round(n * fraction)))
            idx = sorted(rng.sample(range(n), keep))
            out[c] = {
                "unsafe": [d["unsafe"][i] for i in idx],
                "benign": [d["benign"][i] for i in idx],
            }
        return out

    # ------------------------------------------------------------------ #
    # text encoding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _tokenize(self, prompts: Sequence[str]):
        tok = self.pipe.tokenizer
        return tok(
            list(prompts),
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def _encode_pooled(self, prompts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        """phi_bar(.) -> l2-normalized pooled embeddings [N, d]."""
        outs = []
        for i in range(0, len(prompts), batch_size):
            enc = self._tokenize(prompts[i : i + batch_size]).to(self.device)
            o = self.pipe.text_encoder(enc["input_ids"])
            outs.append(o.pooler_output.float())
        e = torch.cat(outs, dim=0)
        return F.normalize(e, dim=-1)

    @torch.no_grad()
    def _encode_tokens(self, prompts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """phi(.) -> token features [N, T, d] and attention mask [N, T]."""
        enc = self._tokenize(prompts).to(self.device)
        # SD-v1.x encodes input_ids only; passing the padding mask changes pad-token features
        h = self.pipe.text_encoder(enc["input_ids"]).last_hidden_state
        return h, enc["attention_mask"]

    @torch.no_grad()
    def _eos_index(self, prompts: Sequence[str]) -> torch.Tensor:
        enc = self._tokenize(prompts)
        return enc["input_ids"].argmax(dim=-1)  # CLIP: eos id is the largest

    # ------------------------------------------------------------------ #
    # (1) prompt routing
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def route(self, prompt: str):
        """Returns (routed_categories, weights[K], per-category selected anchor idx)."""
        cfg = self.cfg
        x = self._encode_pooled([prompt])[0]  # [d], normalized

        scores, sel_idx = {}, {}
        for c in self.categories:
            m = self.u_hat[c] @ x - self.b_hat[c] @ x            # margins m_i(p) [n_c]
            # smooth maximum: gamma * log mean exp(m / gamma)
            s = cfg.route_gamma * (
                torch.logsumexp(m / cfg.route_gamma, dim=0)
                - torch.log(torch.tensor(float(m.numel()), device=m.device))
            )
            scores[c] = s
            L = min(cfg.anchor_topl, m.numel())
            sel_idx[c] = torch.topk(m, k=L).indices
        active = [c for c in self.categories if scores[c].item() > cfg.route_tau]
        if not active:
            return [], torch.empty(0, device=self.device), {}

        active = sorted(active, key=lambda c: -scores[c].item())[: cfg.route_topk]
        s_vec = torch.stack([scores[c] for c in active])
        w = torch.softmax(s_vec, dim=0)
        return active, w, {c: sel_idx[c] for c in active}

    def local_unsafe_normals(self, routed: List[str], sel_idx) -> torch.Tensor:
        """q_c(p) = norm( mean_{i in I_c^L} (u_hat_i - b_hat_i) )  ->  [K, d]."""
        qs = []
        for c in routed:
            idx = sel_idx[c]
            d = (self.u_hat[c][idx] - self.b_hat[c][idx]).mean(dim=0)
            qs.append(F.normalize(d, dim=-1))
        return torch.stack(qs, dim=0)

    # ------------------------------------------------------------------ #
    # (2) text-side correction
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def correct_text(self, H: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor,
                     eos_idx: torch.Tensor) -> torch.Tensor:
        """
        H : [B, T, d] token features
        Q : [K, d]    routed local unsafe normals
        Only tokens violating a routed safe half-space are edited; each is moved to
        the nearest point of the feasible intersection (active-set enumeration).
        """
        cfg = self.cfg
        B, T, d = H.shape
        Hf = H.float().reshape(B * T, d)
        Qf = Q.float()

        m = cfg.lambda_text * Hf.norm(dim=-1)                 # [BT] margin m_t
        viol = (Hf @ Qf.T + m[:, None]) > 0                   # [BT, K]

        editable = mask.reshape(-1).bool()
        if cfg.skip_special_tokens:
            special = torch.zeros(B, T, dtype=torch.bool, device=H.device)
            special[:, 0] = True                              # BOS
            special[torch.arange(B, device=H.device), eos_idx.to(H.device)] = True
            editable = editable & ~special.reshape(-1)
        viol = viol & editable[:, None]

        out = Hf.clone()
        K = Qf.shape[0]
        # group tokens by identical violation pattern -> batched active-set solve
        patterns = torch.unique(viol[viol.any(dim=1)], dim=0)
        for pat in patterns:
            rows = (viol == pat).all(dim=1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            P = pat.nonzero(as_tuple=True)[0]                 # violated categories
            A = Qf[P]                                         # [p, d]
            h = Hf[rows]                                      # [n, d]
            mt = m[rows]                                      # [n]
            best_z = h.clone()
            best_dist = torch.full((h.shape[0],), float("inf"), device=h.device)

            for r in range(1, P.numel() + 1):
                for S in itertools.combinations(range(P.numel()), r):
                    S = list(S)
                    A_S = A[S]                                # [s, d]
                    G = A_S @ A_S.T + cfg.proj_eps * torch.eye(
                        len(S), device=h.device, dtype=h.dtype)
                    rhs = (h @ A_S.T) + mt[:, None]           # [n, s]
                    mu = torch.linalg.solve(G, rhs.T).T       # [n, s]
                    z = h - mu @ A_S                          # [n, d]
                    ok = (mu >= -1e-6).all(dim=1) & (
                        (z @ A.T + mt[:, None]) <= 1e-5).all(dim=1)
                    dist = (z - h).norm(dim=-1)
                    take = ok & (dist < best_dist)
                    best_dist = torch.where(take, dist, best_dist)
                    best_z = torch.where(take[:, None], z, best_z)

            out[rows] = h + cfg.alpha_text * (best_z - h)

        return out.reshape(B, T, d).to(H.dtype)

    # ------------------------------------------------------------------ #
    # ablation: SAFREE-style global pooled projection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_global_subspace(self, rank: int) -> torch.Tensor:
        E = torch.cat([self.u_hat[c] for c in self.categories], dim=0).T  # [d, N]
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
        return U[:, :rank].contiguous()                                    # [d, k]

    @torch.no_grad()
    def _project_out_global(self, H: torch.Tensor) -> torch.Tensor:
        U = self._U_global.to(H.dtype)
        n0 = H.norm(dim=-1, keepdim=True)
        Hp = H - (H @ U) @ U.T
        return Hp * (n0 / (Hp.norm(dim=-1, keepdim=True) + 1e-8))

    # ------------------------------------------------------------------ #
    # (3) residual-side correction
    # ------------------------------------------------------------------ #
    def _branch_prompts(self, prompt: str, routed: List[str], sel_idx):
        neg, pos = [], []
        for c in routed:
            idx = sel_idx[c].tolist()
            u = ", ".join(self.bank[c]["unsafe"][i] for i in idx)
            b = ", ".join(self.bank[c]["benign"][i] for i in idx)
            if self.cfg.residual_branch_mode == "prompt_anchors":
                u, b = f"{prompt}, {u}", f"{prompt}, {b}"
            neg.append(u)
            pos.append(b)
        return neg, pos

    @staticmethod
    def _residual_delta(r: torch.Tensor, g_list: List[torch.Tensor],
                        w: torch.Tensor, cfg: CALMConfig) -> torch.Tensor:
        """Positive-alignment removal + cap. r, g: [B, C, H, W]; alignment taken along the channel dim."""
        r_norm = r.norm(dim=1, keepdim=True)                     # [B,1,H,W]
        delta = torch.zeros_like(r)
        for wc, g in zip(w, g_list):
            g_bar = g / (g.norm(dim=1, keepdim=True) + 1e-8)
            align = (g_bar * r).sum(dim=1, keepdim=True)         # g^T r
            coef = torch.clamp(align - cfg.lambda_res * r_norm, min=0.0)
            delta = delta - cfg.gamma_res * wc * coef * g_bar
        d_norm = delta.norm(dim=1, keepdim=True)
        scale = torch.clamp(cfg.res_cap * r_norm / (d_norm + 1e-8), max=1.0)
        return delta * scale

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, prompt: str, num_images: int = 1, steps: int = 50,
                 guidance_scale: float = 7.5, height: int = 512, width: int = 512,
                 generator: Optional[torch.Generator] = None, verbose: bool = False):
        pipe, cfg = self.pipe, self.cfg
        unet, sched, vae = pipe.unet, pipe.scheduler, pipe.vae
        B = num_images

        # ---- conditioning -------------------------------------------------
        H, mask = self._encode_tokens([prompt])
        eos = self._eos_index([prompt])
        H_uncond, _ = self._encode_tokens([cfg.negative_prompt])

        routed, w, sel_idx = ([], torch.empty(0, device=self.device), {})
        if cfg.ablation == "global_pooled":
            H = self._project_out_global(H)
        else:
            routed, w, sel_idx = self.route(prompt)
            if verbose:
                print(f"[route] {prompt[:70]!r} -> "
                      f"{list(zip(routed, [round(x, 3) for x in w.tolist()])) or 'none (pass-through)'}")
            if routed and cfg.use_text_correction:
                Q = self.local_unsafe_normals(routed, sel_idx)
                H = self.correct_text(H, Q, mask, eos)

        H = H.expand(B, -1, -1).contiguous()
        H_uncond = H_uncond.expand(B, -1, -1).contiguous()

        do_residual = (
            cfg.use_residual_correction and bool(routed) and cfg.ablation == "none"
        )
        H_neg = H_pos = None
        if do_residual:
            neg_p, pos_p = self._branch_prompts(prompt, routed, sel_idx)
            H_neg, _ = self._encode_tokens(neg_p)          # [K, T, d]
            H_pos, _ = self._encode_tokens(pos_p)
            H_neg = H_neg.repeat_interleave(B, dim=0)      # [K*B, T, d]
            H_pos = H_pos.repeat_interleave(B, dim=0)

        # ---- latents ------------------------------------------------------
        ch = unet.config.in_channels
        f = pipe.vae_scale_factor
        latents = torch.randn(
            (B, ch, height // f, width // f),
            generator=generator, device=self.device, dtype=self.dtype)
        sched.set_timesteps(steps, device=self.device)
        latents = latents * sched.init_noise_sigma
        n_steps = len(sched.timesteps)
        lo, hi = int(cfg.res_start_frac * n_steps), int(cfg.res_end_frac * n_steps)
        # stochastic schedulers (DDPM, DDIM with eta) draw step noise from the same generator
        step_kwargs = {}
        if "generator" in inspect.signature(sched.step).parameters:
            step_kwargs["generator"] = generator

        # ---- denoising loop -----------------------------------------------
        for i, t in enumerate(sched.timesteps):
            lat_in = sched.scale_model_input(latents, t)
            in_window = do_residual and (lo <= i < hi)

            embeds = [H_uncond, H]
            if in_window:
                embeds += [H_neg, H_pos]

            model_in = torch.cat(
                [lat_in] * 2 + ([lat_in.repeat(len(routed), 1, 1, 1)] * 2 if in_window else []),
                dim=0)
            eps = unet(model_in, t, encoder_hidden_states=torch.cat(embeds, dim=0)).sample

            eps_u, eps_c = eps[:B], eps[B : 2 * B]
            r = eps_c - eps_u                                     # CFG residual
            if in_window:
                K = len(routed)
                neg = eps[2 * B : 2 * B + K * B].view(K, B, *r.shape[1:])
                pos = eps[2 * B + K * B :].view(K, B, *r.shape[1:])
                g_list = [neg[k] - pos[k] for k in range(K)]
                r = r + self._residual_delta(r.float(), [g.float() for g in g_list],
                                             w, cfg).to(r.dtype)
            eps_hat = eps_u + guidance_scale * r
            latents = sched.step(eps_hat, t, latents, **step_kwargs).prev_sample

        # ---- decode --------------------------------------------------------
        img = vae.decode(latents / vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return pipe.numpy_to_pil(img.permute(0, 2, 3, 1).float().cpu().numpy())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CALM: routed counterfactual correction for safe T2I (SD-v1.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("model / sampling")
    g.add_argument("--model-id", type=str, default="CompVis/stable-diffusion-v1-4")
    g.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim", "pndm"])
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--guidance-scale", type=float, default=7.5)
    g.add_argument("--height", type=int, default=512)
    g.add_argument("--width", type=int, default=512)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])

    g = p.add_argument_group("input / output")
    g.add_argument("--prompt", type=str, default=None)
    g.add_argument("--prompt-file", type=str, default=None,
                   help="benchmark .csv (see --prompt-column), a .json list / list of "
                        "{'prompt': ...}, or a text file with one prompt per line")
    g.add_argument("--num-images", type=int, default=1, help="images per prompt")
    g.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT,
                   help="CFG negative prompt for the unconditional branch")
    g.add_argument("--output-dir", type=str, default="outputs/calm")
    g.add_argument("--route-only", action="store_true",
                   help="print routing decisions without running the diffusion loop")
    g.add_argument("--verbose", action="store_true")

    g = p.add_argument_group("routing")
    g.add_argument("--route-topk", type=int, default=3, help="K")
    g.add_argument("--route-tau", type=float, default=0.0, help="tau_r")
    g.add_argument("--route-gamma", type=float, default=0.1, help="gamma_r (smooth-max temp)")
    g.add_argument("--anchor-topl", type=int, default=5, help="|I_c^L(p)|")

    g = p.add_argument_group("text-side correction")
    g.add_argument("--lambda-text", type=float, default=0.05)
    g.add_argument("--alpha-text", type=float, default=1.0)
    g.add_argument("--proj-eps", type=float, default=1e-6)
    g.add_argument("--no-text-correction", action="store_true")
    g.add_argument("--edit-special-tokens", action="store_true",
                   help="also edit BOS/EOS/pad positions (default: skip)")

    g = p.add_argument_group("residual-side correction")
    g.add_argument("--gamma-res", type=float, default=1.0)
    g.add_argument("--lambda-res", type=float, default=0.05)
    g.add_argument("--res-cap", type=float, default=0.5, help="eta")
    g.add_argument("--res-start-frac", type=float, default=0.0)
    g.add_argument("--res-end-frac", type=float, default=0.3, help="beta")
    g.add_argument("--residual-branch-mode", type=str, default="anchors",
                   choices=["anchors", "prompt_anchors"])
    g.add_argument("--no-residual-correction", action="store_true")

    g = p.add_argument_group("counterfactual bank / ablations")
    g.add_argument("--bank-json", type=str, default=None,
                   help="override the built-in bank: {cat: {unsafe: [...], benign: [...]}}")
    g.add_argument("--bank-fraction", type=float, default=1.0)
    g.add_argument("--bank-seed", type=int, default=42)
    g.add_argument("--ablation", type=str, default="none", choices=["none", "global_pooled"])
    g.add_argument("--global-rank", type=int, default=16)

    g = p.add_argument_group("benchmark csv (--prompt-file *.csv)")
    g.add_argument("--prompt-column", type=str, default="prompt",
                   help="CSV column holding the prompt")
    g.add_argument("--seed-column", type=str, default=None,
                   help="per-row seed column (e.g. evaluation_seed); default: --seed for every row")
    g.add_argument("--guidance-column", type=str, default=None,
                   help="per-row CFG scale column (e.g. evaluation_guidance); "
                        "default: --guidance-scale")
    g.add_argument("--rows", type=str, default=None,
                   help="row slice 'start,end' (0-based, end exclusive) for sharding across GPUs")
    g.add_argument("--overwrite", action="store_true",
                   help="regenerate images that already exist (default: skip, so runs can resume)")
    return p


@dataclass
class Job:
    prompt: str
    name: str        # output file stem
    seed: int
    guidance: float


def _case_number(row: Dict[str, str], row_index: int, path: str) -> int:
    """Image id for a CSV row: case_number column, COCO image_id, else the row index."""
    if row.get("case_number", "").strip():
        return int(float(row["case_number"]))
    if "coco" in os.path.basename(path).lower() and row.get("image_id", "").strip():
        return int(float(row["image_id"]))
    return row_index


def load_csv_jobs(args) -> List[Job]:
    import csv

    with open(args.prompt_file, newline="", encoding="utf-8") as f:
        # leading "# ..." lines are source / provenance comments
        reader = csv.DictReader(itertools.dropwhile(lambda l: l.startswith("#"), f))
        if args.prompt_column not in (reader.fieldnames or []):
            raise SystemExit(f"column '{args.prompt_column}' not in {args.prompt_file}; "
                             f"available: {reader.fieldnames}")
        rows = list(reader)
    start, end = 0, len(rows)
    if args.rows:
        start, end = (int(x) for x in args.rows.split(","))

    jobs = []
    for i in range(start, min(end, len(rows))):
        row = rows[i]
        prompt = row.get(args.prompt_column) or ""
        if not prompt.strip():
            continue
        seed = int(float(row[args.seed_column])) if args.seed_column else args.seed
        guidance = (float(row[args.guidance_column]) if args.guidance_column
                    else args.guidance_scale)
        jobs.append(Job(prompt, str(_case_number(row, i, args.prompt_file)), seed, guidance))
    return jobs


def load_jobs(args) -> Tuple[List[Job], bool]:
    """Returns (jobs, is_benchmark). Benchmark (CSV) runs write <output-dir>/all/<case>.png."""
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file and args.prompt_file.endswith(".csv"):
        return load_csv_jobs(args), True
    elif args.prompt_file and args.prompt_file.endswith(".json"):
        data = json.load(open(args.prompt_file))
        prompts = [d["prompt"] if isinstance(d, dict) else d for d in data]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        raise SystemExit("provide --prompt or --prompt-file")
    return [Job(p, f"{j:05d}", args.seed + j, args.guidance_scale)
            for j, p in enumerate(prompts)], False


def build_config(args) -> CALMConfig:
    return CALMConfig(
        route_topk=args.route_topk,
        route_tau=args.route_tau,
        route_gamma=args.route_gamma,
        anchor_topl=args.anchor_topl,
        use_text_correction=not args.no_text_correction,
        lambda_text=args.lambda_text,
        alpha_text=args.alpha_text,
        proj_eps=args.proj_eps,
        skip_special_tokens=not args.edit_special_tokens,
        use_residual_correction=not args.no_residual_correction,
        gamma_res=args.gamma_res,
        lambda_res=args.lambda_res,
        res_cap=args.res_cap,
        res_start_frac=args.res_start_frac,
        res_end_frac=args.res_end_frac,
        ablation=args.ablation,
        global_rank=args.global_rank,
        bank_fraction=args.bank_fraction,
        bank_seed=args.bank_seed,
        residual_branch_mode=args.residual_branch_mode,
        negative_prompt=args.negative_prompt,
    )


def load_calm(args) -> CALM:
    if args.bank_json:
        COUNTERFACTUAL_BANK.clear()
        COUNTERFACTUAL_BANK.update(json.load(open(args.bank_json)))

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    sched_cls = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler, "pndm": PNDMScheduler}[args.scheduler]
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    for m in (pipe.unet, pipe.vae, pipe.text_encoder):
        m.requires_grad_(False)
    return CALM(pipe, build_config(args))


def main():
    args = build_argparser().parse_args()
    jobs, is_benchmark = load_jobs(args)
    calm = load_calm(args)

    if args.route_only:
        for job in jobs:
            routed, w, _ = calm.route(job.prompt)
            print(json.dumps({"prompt": job.prompt, "routed": routed,
                              "weights": [round(x, 4) for x in w.tolist()]}, ensure_ascii=False))
        return

    # benchmark runs use the <run>/all/<case>.png layout that evaluate.py expects
    img_dir = os.path.join(args.output_dir, "all") if is_benchmark else args.output_dir
    os.makedirs(img_dir, exist_ok=True)

    def filename(job: Job, k: int) -> str:
        stem = job.name if (is_benchmark and args.num_images == 1) else f"{job.name}_{k}"
        return os.path.join(img_dir, f"{stem}.png")

    n_saved = 0
    with open(os.path.join(args.output_dir, "meta.jsonl"), "a") as meta:
        for job in jobs:
            files = [filename(job, k) for k in range(args.num_images)]
            if not args.overwrite and all(os.path.exists(fn) for fn in files):
                continue
            gen = torch.Generator(device=args.device).manual_seed(job.seed)
            images = calm(job.prompt, num_images=args.num_images, steps=args.steps,
                          guidance_scale=job.guidance, height=args.height,
                          width=args.width, generator=gen, verbose=args.verbose)
            routed, w, _ = calm.route(job.prompt)
            for fn, im in zip(files, images):
                im.save(fn)
                meta.write(json.dumps({"file": os.path.basename(fn), "prompt": job.prompt,
                                       "routed": routed,
                                       "weights": [round(x, 4) for x in w.tolist()],
                                       "seed": job.seed, "guidance": job.guidance},
                                      ensure_ascii=False) + "\n")
                n_saved += 1
            meta.flush()
    print(f"saved {n_saved} images -> {img_dir}")


if __name__ == "__main__":
    main()