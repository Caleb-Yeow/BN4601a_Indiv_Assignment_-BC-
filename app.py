"""BN4601A — AI Perception System: Breast Cancer Classification on Mammography.

Gradio Space front-end for a YOLO11s classification model trained on the
CBIS-DDSM mass cases (whole-mammogram, benign vs. malignant).

Held-out test performance (361 images, patient-disjoint from training):
    AUC 0.767 | accuracy 0.701 | sensitivity 0.690 | specificity 0.708  @ 0.50
    AUC 0.767 | accuracy 0.645 | sensitivity 0.786 | specificity 0.551  @ 0.24
"""
import gradio as gr
from ultralytics import YOLO

# Referral threshold on P(malignant). Selected on the VALIDATION set to reach
# >=85% sensitivity, then applied unchanged to the held-out test set. It sits
# below 0.5 on purpose: in screening, a false positive costs a recall
# appointment while a false negative costs a delayed cancer diagnosis, so the
# system is tuned to err toward over-referral.
REFERRAL_THRESHOLD = 0.236

model = YOLO('best.pt')
idx_to_name = model.names
malignant_idx = [i for i, n in idx_to_name.items() if n == 'malignant'][0]


def classify_mammogram(image):
    if image is None:
        return {}, 'Upload a mammogram to see a result.'

    results = model.predict(source=image, verbose=False)
    probs = results[0].probs.data.cpu().numpy()
    scores = {idx_to_name[i]: float(probs[i]) for i in range(len(probs))}

    p_malignant = float(probs[malignant_idx])
    if p_malignant >= REFERRAL_THRESHOLD:
        decision = (
            f'### ⚠️ FLAG FOR REVIEW\n'
            f'P(malignant) = **{p_malignant:.1%}** — at or above the '
            f'{REFERRAL_THRESHOLD:.0%} referral threshold.'
        )
    else:
        decision = (
            f'### ✅ No flag raised\n'
            f'P(malignant) = **{p_malignant:.1%}** — below the '
            f'{REFERRAL_THRESHOLD:.0%} referral threshold.'
        )

    decision += (
        '\n\nThe threshold is deliberately set below 50% so the system errs toward '
        'over-referral, reflecting the asymmetric cost of errors in screening.'
    )
    return scores, decision


demo = gr.Interface(
    fn=classify_mammogram,
    inputs=gr.Image(type='pil', label='Upload Screening Mammogram'),
    outputs=[
        gr.Label(num_top_classes=2, label='Class Probabilities'),
        gr.Markdown(label='Triage Decision'),
    ],
    title='AI Perception System: Breast Cancer Classification on Mammography',
    description=(
        'Upload a full screening mammogram to estimate **Benign** vs. **Malignant**.\n\n'
        'YOLO11s classifier trained on CBIS-DDSM mass cases with a patient-level split. '
        'Held-out test AUC 0.767 (n=361).\n\n'
        '**Educational demonstration only — not a medical device, and not for '
        'diagnosis or clinical decision-making.** (BN4601A)'
    ),
    examples=['example_benign.jpg', 'example_malignant.jpg'],
)

demo.launch()
