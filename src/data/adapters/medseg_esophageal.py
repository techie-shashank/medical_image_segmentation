from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import DatasetAdapter, EvaluationCase


class MedSegEsophagealAdapter(DatasetAdapter):
	"""Adapter for MedSeg esophageal dataset.

	Expected layout:
	- dataset_root/
		- img_0.nii.gz, img_1.nii.gz, ...
		- msk_0.nii.gz, msk_1.nii.gz, ...

	Also supports `.nii` files.
	"""

	@property
	def name(self) -> str:
		return "medseg_esophageal"

	def case_entries(self) -> list[str]:
		"""Extract unique case identifiers from image files."""
		cases: list[str] = []
		for file_path in sorted(self.dataset_root.iterdir()):
			if not file_path.is_file():
				continue

			filename = file_path.name
			if not (filename.startswith("img_") and (filename.endswith(".nii") or filename.endswith(".nii.gz"))):
				continue

			case_id = self._extract_case_id(filename)
			if case_id:
				cases.append(case_id)

		return cases

	@staticmethod
	def _extract_case_id(filename: str) -> str:
		"""Extract numeric/string ID from image filename."""
		if filename.endswith(".nii.gz"):
			stem = filename[: -len(".nii.gz")]
		elif filename.endswith(".nii"):
			stem = filename[: -len(".nii")]
		else:
			return ""

		if not stem.startswith("img_"):
			return ""

		return stem[len("img_") :]

	def _find_existing_file(self, prefix: str, case_id: str) -> Path | None:
		"""Resolve `.nii.gz` first, then `.nii` for a given file prefix and case ID."""
		for ext in (".nii.gz", ".nii"):
			candidate = self.dataset_root / f"{prefix}{case_id}{ext}"
			if candidate.exists():
				return candidate
		return None

	def build_case(self, case_id: str) -> EvaluationCase | None:
		"""Build evaluation case for a given case ID."""
		image_path = self._find_existing_file("img_", case_id)
		mask_path = self._find_existing_file("msk_", case_id)

		if image_path is None or mask_path is None:
			return None

		return EvaluationCase(
			image_path=image_path,
			target_paths={
				"esophageal cancer": mask_path,
			},
		)
