import os


def test_base(app_client):
    response = app_client.get("/")
    assert response.status_code == 200
    assert response.json()["message"].\
        startswith("Welcome to the image classifier")


def test_classify_healthy(app_client):
    response = app_client.post(
        "/classify",
        files={"files": open(os.getcwd() + "/tests/images/healthy.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0][0]["label"] == "Healthy"
    assert response.json()["data"][0][0]["score"] > 0.9


def test_classify_early_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"files": open(os.getcwd() + "/tests/images/early_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0][0]["label"] == "Early blight"
    assert response.json()["data"][0][0]["score"] > 0.9


def test_classify_late_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"files": open(os.getcwd() + "/tests/images/late_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0][0]["label"] == "Late blight"
    assert response.json()["data"][0][0]["score"] > 0.9


def test_all_validation_images(app_client):
    val_images_dir = os.getcwd() + "/tests/images/val_images"
    early_blight_directory = "Early_Blight"
    late_blight_directory = "Late_Blight"
    healthy_directory = "Healthy"

    directories = [early_blight_directory, late_blight_directory, healthy_directory]
    labels = ["Early blight", "Late blight", "Healthy"]

    failed = []
    for didx in range(len(directories)):
        for filename in os.listdir(val_images_dir + "/" + directories[didx]):
            f = os.path.join(val_images_dir + "/" + directories[didx], filename)
            response = app_client.post(
                "/classify",
                files={"files": open(f, "rb")},
            )
            assert response.status_code == 200
            assert response.json()["message"] == "Successful classification"
            max_score = 0.0
            final_label = ""
            for category in response.json()["data"][0]:
                if category["score"] > max_score:
                    max_score = category["score"]
                    final_label = category["label"]
            if final_label != labels[didx]:
                item = {"filename": filename, "response": response.json()["data"][0]}
                failed.append(item)
    with open("test_all_val_images_result.txt", "w") as results_file:
        print("Total failed", len(failed), file=results_file)
        print("Failed images", failed, file=results_file)
