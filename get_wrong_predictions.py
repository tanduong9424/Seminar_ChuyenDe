import json
from pathlib import Path

import torch

from model import TransformerClassifier


def load_vocab(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Model configurations
MODEL_CONFIGS = {
    "1": {"model_file": "model_Transformer_d64_ff128.pt", "d_model": 64, "d_ff": 128, "name": "Transformer #1 (d=64, ff=128)"},
    "2": {"model_file": "model_Transformer_d128_ff256.pt", "d_model": 128, "d_ff": 256, "name": "Transformer #2 (d=128, ff=256)"},
    "3": {"model_file": "model_Transformer_d32_ff64.pt", "d_model": 32, "d_ff": 64, "name": "Transformer #3 (d=32, ff=64)"},
    "4": {"model_file": "model_MLPBaseline_d64.pt", "is_mlp": True, "name": "MLP Baseline"},
}


def get_wrong_predictions(model, test_input_ids, test_labels, test_texts, meta, model_name):
    """Get and display wrong predictions"""
    # Get predictions
    with torch.no_grad():
        logits = model(test_input_ids)
        preds = logits.argmax(dim=-1)
    
    # Find wrong predictions
    wrong_mask = preds != test_labels
    wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
    
    label_names = meta["label_names"]
    
    print(f"\n{'='*100}")
    print(f"Model: {model_name}")
    print(f"Total test samples: {len(test_labels)}")
    print(f"Wrong predictions: {len(wrong_indices)}")
    print(f"{'='*100}")
    
    if len(wrong_indices) == 0:
        print("\n✓ Mô hình dự đoán đúng tất cả các mẫu test!")
        return
    
    for i, idx in enumerate(wrong_indices, 1):
        idx = idx.item()
        text = test_texts[idx]
        true_label = test_labels[idx].item()
        pred_label = preds[idx].item()
        
        print(f"\n[{i}] ID: {idx}")
        print(f"    Text: {text}")
        print(f"    True label: {label_names[true_label]}")
        print(f"    Predicted: {label_names[pred_label]}")


def load_model(model_config, meta, results_dir):
    """Load model based on configuration"""
    model_path = results_dir / model_config["model_file"]
    
    if model_config.get("is_mlp"):
        # Load MLP Baseline
        from train import MLPBaseline
        model = MLPBaseline(
            vocab_size=meta["vocab_size"],
            d_model=model_config.get("d_model", 64),
            num_classes=meta["num_classes"],
        )
    else:
        # Load Transformer
        model = TransformerClassifier(
            vocab_size=meta["vocab_size"],
            d_model=model_config["d_model"],
            d_ff=model_config["d_ff"],
            max_len=meta["max_len"],
            num_classes=meta["num_classes"],
        )
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def show_menu():
    """Display menu and get user choice"""
    print("\n" + "="*100)
    print("LỰA CHỌN MÔ HÌNH ĐỂ XEM CÁC CÂU TRẢ LỜI SAI TRONG BỘ TEST")
    print("="*100)
    for key in ["1", "2", "3", "4", "5"]:
        if key in MODEL_CONFIGS:
            print(f"{key}. {MODEL_CONFIGS[key]['name']}")
        elif key == "5":
            print(f"{key}. Thoát chương trình")
    print("="*100)
    choice = input("Nhập lựa chọn (1-5): ").strip()
    return choice


def main():
    # Paths
    processed_dir = Path("data/processed")
    results_dir = Path("results")
    
    # Load model config
    meta = load_meta(processed_dir / "meta.json")
    
    # Load test data
    test_data = torch.load(processed_dir / "test.pt")
    test_input_ids = test_data["input_ids"]
    test_labels = test_data["labels"]
    test_texts = test_data["texts"]
    
    while True:
        choice = show_menu()
        
        if choice == "5":
            print("\nTạm biệt! 👋")
            break
        
        if choice not in MODEL_CONFIGS:
            print("❌ Lựa chọn không hợp lệ! Vui lòng nhập 1-5.")
            continue
        
        try:
            model_config = MODEL_CONFIGS[choice]
            model = load_model(model_config, meta, results_dir)
            get_wrong_predictions(model, test_input_ids, test_labels, test_texts, meta, model_config["name"])
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            continue


if __name__ == "__main__":
    main()
