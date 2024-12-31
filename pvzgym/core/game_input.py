from pywinauto import Application, keyboard
from pywinauto.timings import Timings

from pvzgym.utils import get_pid

Timings.fast()


class GameInput:

    def __init__(self, process_name: str):
        self.app = Application(backend='uia').connect(process=get_pid(process_name))

    def focus_window(self):
        self.app.top_window().set_focus()

    def click(self, coords: tuple[int, int]):
        self.app.MainDialog.click_input(coords=coords, absolute=False)

    def do_select(self, idx: int):
        '''
        模拟鼠标选择植物
        '''
        first_pos = (220, 135)
        deck_offset = 100
        self.click(coords=(first_pos[0] + deck_offset * idx, first_pos[1]))

    def do_plant(self, x: int, y: int):
        '''
        模拟鼠标种植植物
        '''
        first_pos = (225, 320)
        plant_offset_x = 160
        plant_offset_y = 200
        self.click(coords=(first_pos[0] + plant_offset_x * x, first_pos[1] + plant_offset_y * y))

    def restart_from_menu(self):
        '''
        从游戏菜单中重新开始
        '''
        # 唤起菜单
        keyboard.send_keys('{ESC}')
        # 点击【重新开始】
        self.click(coords=(870, 770))
        # 点击【重来】
        self.click(coords=(700, 770))


if __name__ == '__main__':
    game_input = GameInput('PlantsVsZombies.exe')
