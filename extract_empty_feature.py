import os
import numpy as np
import open_clip

def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
    # ('ViT-H-14', 'laion2b_s32b_b79k'),
    # model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', 'laion2b_s32b_b79k')
    # ('ViT-L-14', 'laion2b_s32b_b82k'),
    # model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', 'laion2b_s32b_b82k')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

    text_tokens = tokenizer(prompts).to(device)
    latent = model.encode_text(text_tokens)

    print(latent.shape)
    c = latent[0].detach().cpu().float().numpy()
    del model
    del tokenizer
    save_dir = f'assets/contexts'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)


if __name__ == '__main__':
    main()




# [('RN50', 'openai'),
# ('RN50', 'yfcc15m'),
# ('RN50', 'cc12m'),
# ('RN50-quickgelu', 'openai'),
# ('RN50-quickgelu', 'yfcc15m'),
# ('RN50-quickgelu', 'cc12m'),
# ('RN101', 'openai'),
# ('RN101', 'yfcc15m'),
# ('RN101-quickgelu', 'openai'),
# ('RN101-quickgelu', 'yfcc15m'),
# ('RN50x4', 'openai'),
# ('RN50x16', 'openai'),
# ('RN50x64', 'openai'),
# ('ViT-B-32', 'openai'),
# ('ViT-B-32', 'laion400m_e31'),
# ('ViT-B-32', 'laion400m_e32'),
# ('ViT-B-32', 'laion2b_e16'),
# ('ViT-B-32', 'laion2b_s34b_b79k'),
# ('ViT-B-32-quickgelu', 'openai'),
# ('ViT-B-32-quickgelu', 'laion400m_e31'),
# ('ViT-B-32-quickgelu', 'laion400m_e32'),
# ('ViT-B-16', 'openai'),
# ('ViT-B-16', 'laion400m_e31'),
# ('ViT-B-16', 'laion400m_e32'),
# ('ViT-B-16', 'laion2b_s34b_b88k'),
# ('ViT-B-16-plus-240', 'laion400m_e31'),
# ('ViT-B-16-plus-240', 'laion400m_e32'),
# ('ViT-L-14', 'openai'),
# ('ViT-L-14', 'laion400m_e31'),
# ('ViT-L-14', 'laion400m_e32'),
# ('ViT-L-14', 'laion2b_s32b_b82k'),
# ('ViT-L-14-336', 'openai'),
# ('ViT-H-14', 'laion2b_s32b_b79k'),
# ('ViT-g-14', 'laion2b_s12b_b42k'),
# ('ViT-bigG-14', 'laion2b_s39b_b160k'),
# ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
# ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
# ('xlm-roberta-large-ViT-H-14','frozen_laion5b_s13b_b90k'),
# ('convnext_base', 'laion400m_s13b_b51k'),
# ('convnext_base_w','laion2b_s13b_b82k'),
# ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
# ('convnext_base_w','laion_aesthetic_s13b_b82k'),
# ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),
# ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),
# ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
# ('convnext_large_d_320', 'laion2b_s29b_b131k_ft'), ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'), ('coca_ViT-B-32', 'laion2b_s13b_b90k'), ('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'), ('coca_ViT-L-14', 'laion2b_s13b_b90k'), ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k')]