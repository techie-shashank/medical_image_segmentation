from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import DatasetAdapter, EvaluationCase


class FedBCaCenter2Adapter(DatasetAdapter):
    """Adapter for FedBCa (Federated Breast Cancer Analysis) Center 2 dataset.
    
    Dataset structure (Center 2):
    - dataset_root/
        - Center2/
            - T2WI/
                - 01.nii.gz, 02.nii.gz, ... (T2-weighted MRI images)
            - Annotation/
                - 01.nii.gz, 02.nii.gz, ... (segmentation masks)
    
    Note: If dataset_root points directly to Center2/ subdirectory, no nesting is needed.
    """

    @property
    def name(self) -> str:
        return "fedbca_center2"

    def _get_data_root(self) -> Path:
        """Get the actual root directory containing T2WI and Annotation subdirs."""
        # Check if dataset_root has nested Center2 directory
        nested_center2 = self.dataset_root / "Center2"
        if nested_center2.exists():
            return nested_center2
        # Otherwise assume dataset_root is already at the correct level
        return self.dataset_root

    def case_entries(self) -> list[str]:
        """Extract unique case identifiers from T2WI directory."""
        data_root = self._get_data_root()
        t2wi_dir = data_root / "T2WI"

        if not t2wi_dir.exists():
            return []

        cases = []
        for img_file in sorted(t2wi_dir.iterdir()):
            if img_file.is_file() and img_file.suffix in [".gz", ".nii"]:
                # Extract case ID (e.g., "01" from "01.nii.gz")
                case_id = img_file.stem
                # Handle double extension (.nii.gz -> stem is .nii -> need stem again)
                if case_id.endswith(".nii"):
                    case_id = Path(case_id).stem
                cases.append(case_id)

        return cases

    def build_case(self, case_id: str) -> EvaluationCase | None:
        """Build evaluation case for a given case ID."""
        data_root = self._get_data_root()
        
        # Find image file (might be .nii or .nii.gz)
        t2wi_dir = data_root / "T2WI"
        annotation_dir = data_root / "Annotation"

        image_path = None
        for candidate in [t2wi_dir / f"{case_id}.nii.gz", t2wi_dir / f"{case_id}.nii"]:
            if candidate.exists():
                image_path = candidate
                break

        # Find annotation file (might be .nii or .nii.gz)
        annotation_path = None
        for candidate in [
            annotation_dir / f"{case_id}.nii.gz",
            annotation_dir / f"{case_id}.nii",
        ]:
            if candidate.exists():
                annotation_path = candidate
                break

        # Both image and annotation must exist
        if image_path is None or annotation_path is None:
            return None

        return EvaluationCase(
            image_path=image_path,
            target_paths={
                "carcinoma": annotation_path,
            },
        )
