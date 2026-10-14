from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_analyze_uses_the_canonical_business_pipeline():
    dataframe = pd.DataFrame(
        {
            "category": ["Electronics", "Furniture", "Electronics"],
            "revenue": [1200, 500, 900],
        }
    )
    content = dataframe.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/analyze",
        files={"file": ("sales.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == 3
    assert payload["analysis"]["status"] in {
        "completed",
        "completed_with_errors",
    }
    assert "insights" in payload["analysis"]
    assert len(payload["data_preview"]) == 3


def test_analyze_accepts_xlsx_uploads():
    dataframe = pd.DataFrame({"category": ["A", "B"], "revenue": [1, 2]})
    buffer = BytesIO()
    dataframe.to_excel(buffer, index=False)

    response = client.post(
        "/analyze",
        files={
            "file": (
                "sales.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["columns"] == 2


def test_analyze_rejects_unsupported_uploads():
    response = client.post(
        "/analyze",
        files={"file": ("sales.json", b"{}", "application/json")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV and XLSX files are supported."
