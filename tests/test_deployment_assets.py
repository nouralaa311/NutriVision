from pathlib import Path


def test_model_weight_file_is_present():
    model_path = Path("nutrivision_deploy/best_model_resnet50_finetune.pth")
    assert model_path.exists(), "Model weights must be present for deployment"
    assert model_path.stat().st_size > 0


def test_nutrition_csv_is_present():
    nutrition_path = Path("nutrivision_deploy/food101_nutrition.csv")
    assert nutrition_path.exists(), "Nutrition csv must be present for deployment"
    assert nutrition_path.stat().st_size > 0
