class PhaseManager:
    """Manages phase transitions (1→2→3) based on epoch number."""

    def __init__(self, config: dict):
        self.phase2_start = config["phases"]["phase2_start"]  # 11
        self.phase3_start = config["phases"]["phase3_start"]  # 26

    def get_phase(self, epoch: int) -> int:
        if epoch >= self.phase3_start:
            return 3
        elif epoch >= self.phase2_start:
            return 2
        return 1

    def has_memory(self, epoch: int) -> bool:
        return self.get_phase(epoch) >= 2

    def has_rnd_fund(self, epoch: int) -> bool:
        return self.get_phase(epoch) >= 2

    def has_rat_mode(self, epoch: int) -> bool:
        return self.get_phase(epoch) >= 3

    def has_scarring(self, epoch: int) -> bool:
        return self.get_phase(epoch) >= 3

    def has_duplicate_filter(self, epoch: int) -> bool:
        return self.get_phase(epoch) >= 3

    def has_venture(self, epoch: int) -> bool:
        return self.get_phase(epoch) == 1
