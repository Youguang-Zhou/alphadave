from dataclasses import dataclass

from pvzgym.core.memory_manipulator import MemoryManipulator
from pvzgym.utils import get_plant_vocab


@dataclass
class Plant:
    id: int
    x: int
    y: int
    hp: int
    hp_norm: int


@dataclass
class Zombie:
    id: int
    x: int
    y: int
    hp: int
    hp_norm: int


class Game:
    def __init__(self, process_name: str):
        self.lane_width = 9  # x
        self.lane_height = 5  # y
        self.garden_size = self.lane_width * self.lane_height
        self.memory = MemoryManipulator(process_name)
        self.vocab = get_plant_vocab()

    @property
    def is_win(self):
        return self.memory.read(offsets=[0x768, 0x560C], type=bool)

    @property
    def is_paused(self):
        return self.memory.read(offsets=[0x768, 0x164], type=bool)

    @property
    def is_terminated(self):
        # TODO: 找到游戏失败的内存地址
        # 用暂停来代替游戏失败（注意：游戏内暂停也会触发 is_terminated = True）
        return self.is_win or self.is_paused

    @property
    def is_selection_phase(self):
        return self.memory.read(offsets=[0x768, 0x15C, 0x2C], type=bool)

    @property
    def sun(self):
        return self.memory.read(offsets=[0x768, 0x5560])

    @property
    def plant_deck(self):
        deck = []
        num_slots = self.memory.read(offsets=[0x768, 0x144, 0x24])
        for i in range(num_slots):
            plant_id = self.memory.read(offsets=[0x768, 0x144, 0x5C + i * 0x50])
            if plant_id != -1 and plant_id < len(self.vocab):
                plant = self.vocab[str(plant_id)]
                plant['is_cooldown_ready'] = self.memory.read(offsets=[0x768, 0x144, 0x70 + i * 0x50], type=bool)
                deck.append(plant)
        return deck

    @property
    def plants(self) -> list[Plant]:
        plant_list = []
        num_plants = self.memory.read(offsets=[0x768, 0xBC])
        for i in range(num_plants):
            plant_list.append(
                Plant(
                    id=self.memory.read(offsets=[0x768, 0xAC, 0x24 + i * 0x204]),
                    x=self.memory.read(offsets=[0x768, 0xAC, 0x28 + i * 0x204]),
                    y=self.memory.read(offsets=[0x768, 0xAC, 0x1C + i * 0x204]),
                    hp=self.memory.read(offsets=[0x768, 0xAC, 0x40 + i * 0x204]),
                    hp_norm=self.memory.read(offsets=[0x768, 0xAC, 0x44 + i * 0x204]),
                )
            )
        return plant_list

    @property
    def zombies(self) -> list[Zombie]:
        zombie_list = []
        num_zombies = self.memory.read(offsets=[0x768, 0xA0])
        for i in range(num_zombies):
            dist = self.memory.read(offsets=[0x768, 0x90, 0x2C + i * 0x204], type=float)
            zombie_list.append(
                Zombie(
                    id=self.memory.read(offsets=[0x768, 0x90, 0x24 + i * 0x204]),
                    x=min(8, int(max(0, dist) // 80)),  # 通过距离计算僵尸所在列数
                    y=self.memory.read(offsets=[0x768, 0x90, 0x1C + i * 0x204]),
                    hp=(
                        # 基础血量
                        self.memory.read(offsets=[0x768, 0x90, 0xC8 + i * 0x204])
                        # 装备1血量
                        + self.memory.read(offsets=[0x768, 0x90, 0xD0 + i * 0x204])
                        # 装备2血量
                        + self.memory.read(offsets=[0x768, 0x90, 0xDC + i * 0x204])
                    ),
                    hp_norm=(
                        # 基础总血量
                        self.memory.read(offsets=[0x768, 0x90, 0xCC + i * 0x204])
                    ),
                )
            )
        return zombie_list

    @property
    def num_selected_plants(self):
        '''
        游戏开始前选植物时的选中的个数
        '''
        try:
            return self.memory.read(offsets=[0x768, 0x8C, 0x774, 0xD24])
        except:
            return 0


if __name__ == '__main__':
    game = Game('PlantsVsZombies.exe')
