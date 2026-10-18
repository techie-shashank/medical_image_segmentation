from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import DatasetAdapter, EvaluationCase


class SKMTEAAdapter(DatasetAdapter):
	"""Adapter for the SKM-TEA knee MRI segmentation dataset.

	Expected layout:
	- dataset_root/
		- images/
			- MTR_001.nii.gz, MTR_002.nii.gz, ...
		- annotations/
			- MTR_001.nii.gz, MTR_002.nii.gz, ...

	The annotation file is a label map with classes:
	1 = patellar cartilage
	2 = femoral cartilage
	3 = medial tibial cartilage
	4 = lateral tibial cartilage
	5 = medial meniscus
	6 = lateral meniscus
	"""

	LABEL_MAP: dict[str, int] = {
		"patellar cartilage": 1,
		"femoral cartilage": 2,
		"medial tibial cartilage": 3,
		"lateral tibial cartilage": 4,
		"medial meniscus": 5,
		"lateral meniscus": 6,
	}

	@property
	def name(self) -> str:
		return "skm_tea"

	def _images_dir(self) -> Path:
		return self.dataset_root / "images"

	def _annotations_dir(self) -> Path:
		return self.dataset_root / "annotations"

	def case_entries(self) -> list[Path]:
		images_dir = self._images_dir()
		if not images_dir.exists():
			return []
		return [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.name.endswith((".nii", ".nii.gz"))]

	@staticmethod
	def _case_stem(path: Path) -> str:
		name = path.name
		if name.endswith(".nii.gz"):
			return name[: -len(".nii.gz")]
		return path.stem

	def build_case(self, case_entry: Path) -> EvaluationCase | None:
		if not case_entry.is_file():
			return None

		case_id = self._case_stem(case_entry)
		annotation_path = self._annotations_dir() / f"{case_id}.nii.gz"

		if not annotation_path.exists():
			return None

		return EvaluationCase(
			image_path=case_entry,
			target_paths={label_name: annotation_path for label_name in self.LABEL_MAP},
			target_labels=self.LABEL_MAP,
			metadata={"case_id": case_id},
		)


class SKMTEAMedialTibialAdapter(SKMTEAAdapter):

	LABEL_MAP: dict[str, int] = {
		# "patellar cartilage": 1,
		# "femoral cartilage": 2,
		"medial tibial cartilage": 3,
		# "lateral tibial cartilage": 4,
		# "medial meniscus": 5,
		# "lateral meniscus": 6,
	}


class SKMTEALateralTibialAdapter(SKMTEAAdapter):

	LABEL_MAP: dict[str, int] = {
		# "patellar cartilage": 1,
		# "femoral cartilage": 2,
		# "medial tibial cartilage": 3,
		"lateral tibial cartilage": 4,
		# "medial meniscus": 5,
		# "lateral meniscus": 6,
	}


class SKMTEAPatellarAdapter(SKMTEAAdapter):

	LABEL_MAP: dict[str, int] = {
		"patellar cartilage": 1,
		# "femoral cartilage": 2,
		# "medial tibial cartilage": 3,
		# "lateral tibial cartilage": 4,
		# "medial meniscus": 5,
		# "lateral meniscus": 6,
	}


class SKMTEAFemoralAdapter(SKMTEAAdapter):

	LABEL_MAP: dict[str, int] = {
		# "patellar cartilage": 1,
		"femoral cartilage": 2,
		# "medial tibial cartilage": 3,
		# "lateral tibial cartilage": 4,
		# "medial meniscus": 5,
		# "lateral meniscus": 6,
	}


class SKMTEAMedialMeniscusAdapter(SKMTEAAdapter):

	LABEL_MAP: dict[str, int] = {
		# "patellar cartilage": 1,
		# "femoral cartilage": 2,
		# "medial tibial cartilage": 3,
		# "lateral tibial cartilage": 4,
		"medial meniscus": 5,
		# "lateral meniscus": 6,
	}


class SKMTEALateralMeniscusAdapter(SKMTEAAdapter):

	LABEL_MAP: dict[str, int] = {
		# "patellar cartilage": 1,
		# "femoral cartilage": 2,
		# "medial tibial cartilage": 3,
		# "lateral tibial cartilage": 4,
		# "medial meniscus": 5,
		"lateral meniscus": 6,
	}