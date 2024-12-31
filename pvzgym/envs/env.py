from dataclasses import dataclass
from itertools import chain

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pvzgym.core.game import Game
from pvzgym.core.game_input import GameInput


@dataclass
class Config:
    max_sun: int = 9990  # 阳光最大值
    max_slot_num: int = 6  # 最大卡槽格数（最大值为14）
    max_zombie_num: int = 16384  # 最大僵尸数量
    process_name: str = 'PlantsVsZombies.exe'


@dataclass
class RewardState:
    attack_plant_hp: int = 0
    defense_plant_hp: int = 0
    zombie_hp: int = 0


class PvZEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=Config):

        self.config = config

        self.game = Game(config.process_name)
        self.game_input = GameInput(config.process_name)

        self.observation_space = spaces.Dict(
            {
                # 阳光
                'sun': spaces.Box(low=0, high=1),
                # 每块地的植物id，-1表示未种植任何植物
                'plant_id': spaces.MultiDiscrete(
                    nvec=self.game.garden_size * [config.max_slot_num + 1],
                    start=self.game.garden_size * [-1],
                ),
                # 每块地的植物血量
                'plant_hp': spaces.Box(low=0, high=1, shape=(self.game.garden_size,)),
                # 每块地的僵尸总个数
                'zombie_num': spaces.MultiDiscrete(self.game.garden_size * [config.max_zombie_num]),
                # 每块地的僵尸总血量
                'zombie_hp': spaces.Box(low=0, high=config.max_zombie_num, shape=(self.game.garden_size,)),
                # 当前植物是否可种植
                'action_ready': spaces.MultiBinary(config.max_slot_num),
            }
        )

        self.action_space = spaces.Discrete(self.game.garden_size * config.max_slot_num + 1)

    @property
    def obs(self):
        # 阳光
        obs_sun = [self.game.sun / self.config.max_sun]
        # 植物
        obs_plants_id = [-1] * self.game.garden_size
        obs_plants_hp = [0] * self.game.garden_size
        for plant in self.game.plants:
            # 排除阳光豆，避免真正要种的植物被覆盖
            if plant.id != 1:
                obs_plants_id[plant.x + plant.y * self.game.lane_width] = plant.id
                obs_plants_hp[plant.x + plant.y * self.game.lane_width] = plant.hp / plant.hp_norm
        # 僵尸
        obs_zombies_num = [0] * self.game.garden_size
        obs_zombies_hp = [0] * self.game.garden_size
        for zombie in self.game.zombies:
            # 有时候 zombie.hp_norm 会返回0
            if zombie.hp_norm != 0:
                obs_zombies_num[zombie.x + zombie.y * self.game.lane_width] += 1
                obs_zombies_hp[zombie.x + zombie.y * self.game.lane_width] += zombie.hp / zombie.hp_norm
        # 是否可以种植
        obs_action_ready = [0] * self.config.max_slot_num
        for i, plant in enumerate(self.game.plant_deck):
            if plant['is_cooldown_ready'] and plant['cost'] <= self.game.sun:
                obs_action_ready[i] = 1
        return {
            'sun': np.array(obs_sun, dtype=np.float32),
            'plant_id': np.array(obs_plants_id),
            'plant_hp': np.array(obs_plants_hp, dtype=np.float32),
            'zombie_num': np.array(obs_zombies_num),
            'zombie_hp': np.array(obs_zombies_hp, dtype=np.float32),
            'action_ready': np.array(obs_action_ready),
        }

    @property
    def info(self):
        return {
            'is_win': self.game.is_win,
            'action_mask': self.get_action_mask(),
        }

    @property
    def reward_state(self):
        return RewardState(
            # 在场攻击性植物的总血量（e.g. 排除阳光豆，阳光炸弹，火炬坚果）
            attack_plant_hp=np.sum([plant.hp for plant in self.game.plants if plant.id not in [1, 2, 3]]),
            # 在场防御性植物的总血量（e.g. 火炬坚果）
            defense_plant_hp=np.sum([plant.hp for plant in self.game.plants if plant.id in [3]]),
            # 在场僵尸的总血量
            zombie_hp=np.sum([zombie.hp for zombie in self.game.zombies]),
        )

    def reset(self, seed=None, options=None):
        # 聚焦游戏窗口
        self.game_input.focus_window()
        # 查看当前游玩结果
        if self.game.is_win:
            # 如果是胜利，则选择从菜单中重置
            self.game_input.restart_from_menu()
        else:
            # 如果是失败，则点击【再次尝试】
            self.game_input.click(coords=(870, 785))
        # 点击【重选上次卡牌】
        while self.game.num_selected_plants == 0:
            # 多点几次有时候会点不到
            self.game_input.click(coords=(1350, 250))
        # 点击【一起摇滚吧！】
        while self.game.is_selection_phase:
            # 多点几次有时候会点不到
            self.game_input.click(coords=(1000, 1200))
        # [选卡界面的僵尸] -> [] -> [第一波僵尸]
        while len(self.game.zombies) != 0:
            # 等待卡槽初始化完毕
            pass
        while len(self.game.zombies) == 0:
            # 等待第一波僵尸出现
            pass
        return self.obs, self.info

    def step(self, action):
        # 获取当前状态
        prev_reward_state = self.reward_state
        # action = 0 表示什么都不做
        if action > 0:
            # 获取当前要种植的植物索引以及种植坐标
            idx, (x, y) = self.parse_action(action)
            # 检查行为合法性
            if self.get_action_mask()[action]:
                # 选择植物 -> 种植植物
                self.game_input.do_select(idx)
                self.game_input.do_plant(x, y)
            else:
                raise Exception(f'Plant {idx} in ({x}, {y}) is illegal!')
        # 当没有可用操作时，等待植物冷却
        while not any(self.obs['action_ready']):
            if self.game.is_terminated:
                break
        # 准备返回值
        observation = self.obs
        reward = self.compute_reward(prev_reward_state, self.reward_state)
        terminated = self.game.is_terminated
        truncated = False
        info = self.info
        return observation, reward, terminated, truncated, info

    def parse_action(self, action):
        '''
        从 action 中获取当前要种植的植物索引以及种植坐标

        e.g.
            |0  | , |0,0,0,0, ..., 0| , |0, 1, ..., 0, 0| , |...|
            |a_0| , |plant_deck_size| , |plant_deck_size| , |plant_deck_size|
        '''
        # action = 0 表示什么都不做
        if action == 0:
            raise Exception('Can not parse action = 0!')
        # 减去action = 0的时候
        action -= 1
        # 当前种植的地块
        pos_to_plant = action // self.config.max_slot_num
        # 当前种植的植物卡组索引
        idx = action - self.config.max_slot_num * pos_to_plant
        # 当前种植的植物具体坐标
        y = pos_to_plant // self.game.lane_width
        x = pos_to_plant - y * self.game.lane_width
        return idx, (x, y)

    def action_to_plant_id(self, action):
        idx, _ = self.parse_action(action)
        return self.game.plant_deck[idx]['id']

    def get_action_mask(self):
        # 先排除 action = 0
        mask = np.zeros(self.action_space.n - 1, dtype=bool)
        # 检查当前土地是否已种植（-1 表示未种植）
        empty_cells = np.where(self.obs['plant_id'] == -1)[0]
        empty_cells = list(chain(*[list(range(i * 6, (i + 1) * 6)) for i in empty_cells]))
        mask[empty_cells] = True
        # 检查当前植物是否可种植（卡槽高亮显示）
        available_plants = np.tile(self.obs['action_ready'], self.game.garden_size).astype(bool)
        mask = np.logical_and(mask, available_plants)
        # 处理阳光豆的特殊情况
        for plant in self.game.plants:
            if plant.id == 1:
                cell_idx = plant.x + plant.y * self.game.lane_width
                plant_idx = np.where(np.array([p['id'] for p in self.game.plant_deck]) == plant.id)[0][0]
                mask[cell_idx * self.config.max_slot_num + plant_idx] = False
        # 最后加入 action = 0
        mask = np.array([True, *mask]).astype(np.int8)
        return mask

    @staticmethod
    def compute_reward(prev: RewardState, curr: RewardState):
        # 攻击性植物减少的血量（越少越好）
        attack_plant_hp_reward = 0
        if (prev.attack_plant_hp - curr.attack_plant_hp) > 0:
            attack_plant_hp_reward = (prev.attack_plant_hp - curr.attack_plant_hp) / prev.attack_plant_hp
            attack_plant_hp_reward = -1 * attack_plant_hp_reward
        # 防御性植物减少的血量（越多越好）
        defense_plant_hp_reward = 0
        if (prev.defense_plant_hp - curr.defense_plant_hp) > 0:
            defense_plant_hp_reward = (prev.defense_plant_hp - curr.defense_plant_hp) / prev.defense_plant_hp
        # 僵尸减少的血量（越多越好）
        zombie_hp_reward = 0
        if (prev.zombie_hp - curr.zombie_hp) > 0:
            zombie_hp_reward = (prev.zombie_hp - curr.zombie_hp) / prev.zombie_hp
        return attack_plant_hp_reward + defense_plant_hp_reward + zombie_hp_reward


if __name__ == '__main__':

    env = gym.make('PlantsVsZombies')
    env.reset()

    while True:
        action_mask = env.unwrapped.get_action_mask()
        action = env.action_space.sample(action_mask)
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
