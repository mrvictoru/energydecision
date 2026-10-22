# Outreach emails (drafts)

Two separate emails. Send the endorsement request only if arXiv asks for a code;
send the supervision enquiry when you are ready. You can combine them if he
replies, but do not open with two asks.

---

## 1. arXiv endorsement request (short)

**Subject:** arXiv endorsement code — cs.LG (first submission)

Dear Professor Halgamuge,

I am preparing an arXiv preprint on planner-distilled Decision Transformers for
multi-market battery dispatch in the NEM (energy plus eight FCAS markets, with a
degradation-aware objective). arXiv requires an endorsement for a first
submission to cs.LG, and has given me this code:

**CODE: [paste the code arXiv shows you]**

If you are willing, please enter it at https://arxiv.org/auth/endorse (or the
link arXiv provides). It takes about a minute. I am happy to send the PDF first
so you can see what it is.

With thanks,
Hou Nam U
victor.u@outlook.com
https://github.com/mrvictoru/energydecision

---

## 2. PhD supervision enquiry

**Subject:** PhD enquiry — offline RL / planner distillation for multi-market
battery dispatch

Dear Professor Halgamuge,

I am an independent researcher (recent graduate) working on offline
reinforcement learning for grid-scale battery dispatch. I am writing to ask
whether you are taking PhD students for the next round, and whether my direction
would fit your group.

**The work.** I study whether an offline sequence model — a Decision Transformer
— can act as a single controller for multi-market battery dispatch: energy
arbitrage plus eight FCAS markets, with battery wear in the reward. Training on
trajectories from online RL policies stalls at a behaviour-cloning ceiling: the
logged policies bid little into FCAS, and no amount of data scaling, prompting,
loss re-weighting, or fine-tuning closes the gap. I break the ceiling by
distilling an honest, non-clairvoyant SDP planning teacher into a standalone
Decision Transformer, conditioned on a state-dependent return-to-go prompt
derived from the planner's cost-to-go. The teacher follows the SDP-with-forecasts
and degradation lineage of your work with Abdulla et al. (IEEE TSG, 2018), and the
paper's degradation model follows Kampker et al. (Batteries, 2025).

**Headline results (simulator, NEM data).** The distilled policy beats a tuned
PPO baseline on four held-out surfaces — 3.84x, 1.80x, 6.70x, and 6.39x — with
all four paired 95% CIs excluding zero, and it passes a piecewise-merit-order
market-impact gate on 9/9 grid-scale cells, with no solver in the control loop
(the value table is precomputed). The main open question is sim-to-real transfer.

**What is released.** Code, training corpora, model weights, and a provenance
record mapping every headline number to the artifact that produced it:
https://github.com/mrvictoru/energydecision
https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp

**The ask.** I am applying in the December scholarship round and would be
grateful to know whether you are supervising in this area, and whether a short
conversation would be possible. The preprint is attached; it is also on arXiv
(pending endorsement): [arXiv ID once available].

Best regards,
Hou Nam U
victor.u@outlook.com
