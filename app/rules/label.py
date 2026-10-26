from dataclasses import dataclass, field


@dataclass
class LabelData:
    claims: list[str] = field(default_factory=list)
    ingredients: list[str] = field(default_factory=list)
    allergen_statement: str | None = None
    protein_g: float | None = None
    gluten_ppm: float | None = None   # lab test result, if one exists
