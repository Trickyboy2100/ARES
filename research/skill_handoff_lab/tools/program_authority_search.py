"""Deduplicate existing search points after declared-authority projection.

A deterministic baseline correction, not a learned mechanism. The original
SearchControl is untouched; its progress, stall, advance and state updates are
used verbatim. Instantiate a new controller per episode/declared family.
"""

def unique_projected_points(points, authority, family):
    if authority not in {'full', 'y_only', 'task'}:
        raise ValueError(f'Unknown authority: {authority}')
    if family not in {'Y', 'YZ'}:
        raise ValueError(f'Unknown declared family: {family}')
    project_z = authority == 'y_only' or (authority == 'task' and family == 'Y')
    unique = []
    for y, z in points:
        point = (y, 0. if project_z else z)
        if point not in unique:
            unique.append(point)
    if not unique:
        raise ValueError('Search requires at least one legal point')
    return unique


class AuthoritySearchControl:
    def __init__(self, authority, family, base_factory=None):
        if base_factory is None:
            # Lazy import keeps this helper independently CPU-testable.
            from program_adaptive_contact import SearchControl
            base_factory = SearchControl
        self.base = base_factory()
        self.original_points = tuple(self.base.points)
        self.base.points = unique_projected_points(self.original_points, authority, family)
        self.authority, self.family = authority, family

    @property
    def points(self):
        return tuple(self.base.points)

    def __call__(self, obs):
        return self.base(obs)
