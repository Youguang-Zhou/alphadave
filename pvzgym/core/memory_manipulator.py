from pymem import Pymem

from pvzgym.utils import get_pid

BASE_ADDRESS = 0x6A9EC0


class MemoryManipulator:

    def __init__(self, process_name: str, base_address: int = BASE_ADDRESS):
        self.base_address = base_address
        self.pymem = Pymem()
        self.pymem.open_process_from_id(process_id=get_pid(process_name))

    def _resolve_offsets(self, offsets: list[int]):
        address = self.pymem.read_long(self.base_address)
        for offset in offsets[:-1]:
            address = self.pymem.read_long(address + offset)
        address += offsets[-1]
        return address

    def read(self, offsets: list[int], type=int):
        address = self._resolve_offsets(offsets)
        if type == int:
            return self.pymem.read_long(address)
        if type == bool:
            return self.pymem.read_bool(address)
        if type == float:
            return self.pymem.read_float(address)
        raise Exception(f'Type {type} not support!')


if __name__ == '__main__':
    memory = MemoryManipulator('PlantsVsZombies.exe')
