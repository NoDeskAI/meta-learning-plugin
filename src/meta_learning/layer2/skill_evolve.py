from __future__ import annotations

from pathlib import Path

from meta_learning.shared.llm import LLMInterface
from meta_learning.shared.models import (
    MetaLearningConfig,
    SkillEvolveResult,
    SkillUpdateAction,
    TaxonomyEntry,
)


class SkillEvolver:
    def __init__(self, config: MetaLearningConfig, llm: LLMInterface) -> None:
        self._config = config
        self._llm = llm

    async def evolve_from_taxonomy(
        self, new_entries: list[TaxonomyEntry]
    ) -> list[SkillEvolveResult]:
        if not new_entries:
            return []

        results: list[SkillEvolveResult] = []
        for entry in new_entries:
            existing_content = self._find_matching_skill(entry)
            result = await self._llm.evaluate_skill_update(entry, existing_content)

            if result.action != SkillUpdateAction.NONE:
                self._apply_skill_update(result)

            results.append(result)

        return results

    def _find_matching_skill(self, entry: TaxonomyEntry) -> str | None:
        skills_dir = Path(self._config.skills_path)
        if not skills_dir.exists():
            return None

        entry_name_normalized = entry.name.lower().replace(" ", "-")
        entry_keywords = {kw.lower() for kw in entry.keywords}

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            content = skill_md.read_text()
            dir_name = skill_dir.name.lower()

            if f"Taxonomy ID: {entry.id}" in content:
                return content

            if dir_name == entry_name_normalized:
                return content

            if _keyword_overlap(dir_name, entry_keywords):
                return content

        return None

    def _apply_skill_update(self, result: SkillEvolveResult) -> None:
        if result.action == SkillUpdateAction.NONE or not result.new_content:
            return

        skills_dir = Path(self._config.skills_path)
        skills_dir.mkdir(parents=True, exist_ok=True)

        if result.action == SkillUpdateAction.CREATE and result.target_skill:
            skill_dir = skills_dir / result.target_skill
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(result.new_content)

        elif result.action == SkillUpdateAction.APPEND and result.target_skill:
            skill_dir = skills_dir / result.target_skill
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                existing = skill_file.read_text()
                skill_file.write_text(existing + result.new_content)

        elif result.action == SkillUpdateAction.REPLACE and result.target_skill:
            skill_dir = skills_dir / result.target_skill
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(result.new_content)


def _keyword_overlap(dir_name: str, keywords: set[str]) -> bool:
    dir_words = set(dir_name.replace("-", " ").replace("_", " ").split())
    return bool(dir_words & keywords)
