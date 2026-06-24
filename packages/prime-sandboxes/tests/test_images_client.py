from prime_sandboxes import BuildImageResponse, ImageClient, ImageVisibility


def test_image_client_transfer_image_payload_and_response():
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return {
                "build_id": "build-123",
                "buildIds": ["build-123"],
                "upload_url": None,
                "fullImagePath": "team-team1/ubuntu:22.04",
                "visibility": "PUBLIC",
            }

    client = ImageClient(DummyAPIClient())
    response = client.transfer_image(
        "ubuntu:22.04",
        image_name="ubuntu",
        image_tag="22.04",
        team_id="team1",
        visibility=ImageVisibility.PUBLIC,
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/images/build"
    assert captured["json"] == {
        "image_name": "ubuntu",
        "image_tag": "22.04",
        "dockerfile_path": "Dockerfile",
        "source_image": "ubuntu:22.04",
        "platform": "linux/amd64",
        "team_id": "team1",
        "visibility": ImageVisibility.PUBLIC,
    }
    assert isinstance(response, BuildImageResponse)
    assert response.build_id == "build-123"
    assert response.build_ids == ["build-123"]
    assert response.upload_url is None
    assert response.full_image_path == "team-team1/ubuntu:22.04"
