from fastapi import HTTPException
import requests
from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile

MOMA_URL = "https://datagems-dev.scayle.es/dmm/api/v1/dataset/search"

class MomaDataset:
    def __init__(self, user_token: str):
        self.external_id = None
        self.user_token = user_token

    def _moma_get(
        self, endpoint: str = MOMA_URL, params: dict | None = None
    ) -> dict:
        """Generalized GET request to MOMA with user token attached."""
        try:
            resp = requests.get(
                endpoint,
                params=params,
                headers={"Authorization": f"Bearer {self.user_token}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data

        except requests.RequestException as e:
            print(f"Request to MOMA failed: {e}")
            raise HTTPException(502, "Request to MOMA failed.")

        except requests.exceptions.JSONDecodeError:
            print("Invalid JSON response from MOMA.")
            raise HTTPException(502, "Invalid response from MOMA.")

    def get_from_external(self, id: str):
        self.external_id = id
        properties = self.get_dataset_properties(id)
        self.name = properties.get("name", "")
        self.metadata = " ".join(
            x if isinstance(x, str) else " ".join(x)
            for x in properties.values()
            if x is not None
        )

    def get_dataset_properties(self, id: str, properties=None) -> dict:
        if properties is None:
            properties = [
                "description",
                "headline",
                "keywords",
                "fieldOfScience",
                "name",
            ]

        data = self._moma_get(
            MOMA_URL, params={"nodeIds": [id], "properties": properties}
        )
        return data["datasets"][0]["nodes"][0]["properties"]

    def get_all_external_ids(self):
        data = self._moma_get(MOMA_URL, params={"properties": ["name"]})
        return [x["nodes"][0]["id"] for x in data["datasets"]]

    def to_dataset_profile(self) -> DatasetProfile:
        """
        Converts the raw metadata fetched from DataGEMS/MOMA 
        into the standard DatasetProfile format used by the recsys.
        """
        # Ensure metadata is a dict
        props = self.metadata if isinstance(self.metadata, dict) else {}
        
        # Extract fields following the logic in fetch_gems_datasets.py
        title = self.name or props.get("name", "")
        headline = props.get("dg:headline", "")
        
        # Handle keywords (ensuring it's a string)
        kw = props.get("dg:keywords", "")
        if isinstance(kw, list):
            keywords = ", ".join(filter(None, kw))
        else:
            keywords = str(kw)

        return DatasetProfile(
            id=self.dataset_id,
            title=title.strip(),
            headline=headline.strip() if headline.lower() != title.lower() else "",
            description=props.get("description", "").strip(),
            keywords=keywords,
            field_of_science=str(props.get("dg:fieldOfScience", "")),
            catalog_summary=""  # This will be filled by the LLM enrichment step
        )