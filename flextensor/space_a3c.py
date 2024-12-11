import tvm
import math
import numpy as np
from functools import reduce
from flextensor.utils import (assert_print, gen_enum, any_factor_split, get_factor_lst, gen_group,
    is_power_of_x)


def able_inline(op, down_graph):
    is_compute = isinstance(op, tvm.te.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    is_output = False
    for i in range(op.num_outputs):
        if op.output(i) not in down_graph:
            is_output = True
            break
    return is_compute and (not has_reduce) and (not is_output)


# class SubSpace(object):
#     def __init__(self, entities):
#         assert_print(isinstance(entities, (list, tuple)) and len(entities) > 0)
#         self.entities = entities
#         self.begin = 0
#         self.end = len(self.entities)

#     def get_entity(self, p):
#         if len(self.entities) < 1:
#             raise RuntimeError("Query from empty space")
#         if 0 <= p < self.end:
#             return self.entities[p]
#         else:
#             raise RuntimeError("Space pointer out of range")

#     def range(self, p, left, right=None):
#         if right is None:
#             right = left
#         left = p - left if p - left >= 0 else 0
#         right = p + right if p + right <= self.end else self.end
#         return range(left, right), self.entities[left:right]

#     def __len__(self):
#         return self.end


class Space(object):
    def __init__(self):
        self.subspaces = {}
        self.types = {}
        self.valid_type_keys = ["fuse", "spatial", "reduce", "reorder", "inline", "unroll", "merge", "special"]
        for type_key in self.valid_type_keys:
            self.types[type_key] = []
        self.dim = 0
        # self.action_map = None

    def add_subspace(self, name, subspace, type_key, override=False):
        if name in self.subspaces and not override:
            raise RuntimeError("Same subspace name")
        assert_print(type_key in self.valid_type_keys)
        self.subspaces[name] = subspace
        self.types[type_key].append(name)
        self.dim += subspace.dim

    def items(self):
        return self.subspaces.items()

    def __len__(self):
        ret = 1
        for _, subspace in self.subspaces.items():
            ret *= len(subspace)
        return ret

    def length(self):
        ret = {}
        total = 1
        added = 0
        for name, subspace in self.subspaces.items():
            ret[name] = len(subspace)
            total *= ret[name]
            added += ret[name]
        ret["total"] = total
        ret["added"] = added
        return ret
    
    # def build_action_map(self):
    #     # 전체 action space를 만들기 위해 각 subspace의 num_direction을 합칩니다.
    #     # action = 0부터 시작해서 subspace별 direction 범위를 설정
    #     if self.action_map is not None:
    #         return
    #     self.action_map = []
    #     for name, subspace in self.subspaces.items():
    #         for d_id in range(subspace.num_direction):
    #             self.action_map.append((name, d_id))

    # @property
    # def action_dim(self):
    #     # 모든 subspace들의 num_direction 합
    #     total = 0
    #     for subspace in self.subspaces.values():
    #         total += subspace.num_direction
    #     return total

    # def get_initial_state_indices(self):
    #     # return list of indices for each subspace
    #     state_indices = []
    #     for name, subspace in self.subspaces.items():
    #         idx = np.random.randint(0, len(subspace))
    #         state_indices.append(idx)
    #     return state_indices

    # def get_state_vector(self, state_indices):
    #     # state_indices: list of int, each int is index chosen in each subspace
    #     # we concat all chosen entities into one vector
    #     vec = []
    #     subspace_names = list(self.subspaces.keys())
    #     for i, name in enumerate(subspace_names):
    #         subspace = self.subspaces[name]
    #         entity = subspace.get_entity(state_indices[i])  # this is a list/array representing the chosen entity
    #         vec.extend(entity)
    #     return np.array(vec, dtype=np.float32)

    # def take_action(self, state_indices, action):
    #     # action을 (subspace, direction)으로 디코딩
    #     if self.action_map is None:
    #         self.build_action_map()
    #     if action < 0 or action >= self.action_dim:
    #         raise RuntimeError("Invalid action")

    #     subspace_name, direction_id = self.action_map[action]
    #     subspace = self.subspaces[subspace_name]

    #     # state는 각 subspace별 index를 담고있다고 가정
    #     # subspace_name의 index를 찾아서 next_entity 호출
    #     # subspace_name이 self.subspaces의 몇 번째인지 알 필요가 있으니
    #     # subspace_name 순서가 필요하다. items() 순서에 의존하기 싫다면
    #     # dict를 리스트로 만들어서 인덱싱하는 방식을 쓸 수도 있음
    #     # 여기서는 그냥 items 순서에 의존한다고 가정(또는 subspace_name->index map 필요)

    #     # subspace_name -> index 맵핑
    #     # 한번만 계산하도록 캐싱할 수도 있음
    #     name_list = list(self.subspaces.keys())
    #     subspace_idx = name_list.index(subspace_name)

    #     current_index = state_indices[subspace_idx]
    #     direction = subspace.get_direction(direction_id)
    #     next_index = subspace.next_entity(current_index, direction)

    #     # 다음 상태 업데이트
    #     next_state_indices = state_indices.copy()
    #     next_state_indices[subspace_idx] = next_index

    #     # reward, done은 여기서는 예시로 0, False
    #     # 실제로는 해당 state에 대한 성능(예: schedule 평가 결과)을 반영해야 함
    #     reward = 0.0
    #     done = False

    #     return next_state_indices, reward, done


DirectedSubSpaceTypeKeys = ["spatial", "reduce"]
UndirectedSubSpaceTypeKeys = ["fuse", "reorder", "unroll", "inline", "merge", "special"]


class SubSpace(object):
    def __init__(self):
        self.dim = 0
        self.static_entities = []
        self.size = 0
        self.num_direction = 0

    def random_entity(self):
        return np.random.choice(self.static_entities)

    def next_entity(self, *args, **kwargs):
        raise NotImplementedError()

    def get_entity(self, p):
        return self.static_entities[p]

    def get_direction(self, num):
        raise NotImplementedError()

    def __len__(self):
        return self.size


class SplitSpace(SubSpace):
    def __init__(self, dim, total, allow_non_divisible='off'):
        super(SplitSpace, self).__init__()
        self.total = total
        self.allow_non_divisible = allow_non_divisible
        self.dim = dim
        self.static_entities = any_factor_split(total, dim, allow_non_divisible=allow_non_divisible)
        self.size = len(self.static_entities)
        self.num_direction = dim * (dim - 1)
        self.directions = []
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    self.directions.append((i, j))
        self.type_key = "split"
    
    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            next_pos = (pos + d[0]) % self.size
            return next_pos
        elif len(d) == 2:
            asc_pos, dec_pos = d[0], d[1]
            assert_print(0 <= asc_pos < self.dim)
            assert_print(0 <= dec_pos < self.dim)
            assert_print(asc_pos != dec_pos)
            current = self.static_entities[pos]
            ret = current.copy()
            left = current[asc_pos] * current[dec_pos]
            canout = False
            next_pos = -1
            while not canout:
                tmp = ret[asc_pos] + 1
                while tmp <= left:
                    if self.allow_non_divisible == 'continuous':
                        break
                    elif self.allow_non_divisible == 'power2' and is_power_of_x(2, tmp):
                        break
                    elif left % tmp == 0:
                        break
                    tmp += 1
                tmp = min(tmp, left)
                ret[asc_pos] = tmp
                ret[dec_pos] = math.ceil(left / tmp)
                try:
                    next_pos = self.static_entities.index(ret)
                    canout = True
                except ValueError:
                    canout = False
            return next_pos
        else:
            raise RuntimeError(
                "Not support for direction more than two dims: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class FuseSpace(SubSpace):
    def __init__(self, dim, elements):
        self.dim = dim
        self.static_entities = gen_group(elements, most_groups=self.dim)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "fuse"
    
    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    
class ReorderSpace(SubSpace):
    def __init__(self, num_spatial_axis):
        self.dim = 1
        self.static_entities = [[i] for i in range(num_spatial_axis)]
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "reorder"
    
    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class UnrollSpace(SubSpace):
    def __init__(self, steps, explicit=False):
        super(UnrollSpace, self).__init__()
        self.dim = 2
        self.static_entities = []
        self.steps = steps
        explicits = [1] if explicit else [0, 1]
        for step in steps:
            for _explicit in explicits:
                self.static_entities.append([step, _explicit])
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "unroll"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class PosSpace(SubSpace):
    def __init__(self, parts, num_axis):
        self.dim = 2
        self.static_entities = []
        self.parts = parts
        self.num_axis = num_axis
        for i in range(parts):
            for j in range(num_axis):
                self.static_entities.append([i, j])
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "local"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class InlineSpace(SubSpace):
    def __init__(self, inline_op_pos, op_num, force_inline=False):
        self.dim = op_num
        self.static_entities = []
        self.able_inline_list = inline_op_pos
        if force_inline:
            entity = [0] * op_num
            for pos in inline_op_pos:
                entity[pos] = 1
            self.static_entities.append(entity)
        else:
            num_inline_ops = len(inline_op_pos)
            enums = gen_enum([1, 0], num_inline_ops)
            for enum in enums:
                entity = [0] * op_num
                for i in range(num_inline_ops):
                    entity[inline_op_pos[i]] = enum[i]
                self.static_entities.append(entity)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "inline"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    def able_inline(self, pos):
        return pos in self.able_inline_list


class MergeSpce(SubSpace):
    def __init__(self, merge_op_pos, op_num, force_merge=False):
        self.dim = op_num
        self.static_entities = []
        self.able_merge_list = merge_op_pos
        if force_merge:
            entity = [0] * op_num
            for pos in merge_op_pos:
                entity[pos] = 1
            self.static_entities.append(entity)
        else:
            num_merge_ops = len(merge_op_pos)
            enums = gen_enum([1, 0], num_merge_ops)
            for enum in enums:
                entity = [0] * op_num
                for i in range(num_merge_ops):
                    entity[merge_op_pos[i]] = enum[i]
                self.static_entities.append(entity)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "merge"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    def able_merge(self, pos):
        return pos in self.able_merge_list


class EnumSpace(SubSpace):
    def __init__(self, knobs):
        self.dim = 2
        self.static_entities = knobs
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


def generate_inline_space(op_lst, down_graph, force_inline=False):
    inline_op_pos = []
    for i, op in enumerate(op_lst):
        if able_inline(op, down_graph):
            inline_op_pos.append(i)
    return InlineSpace(inline_op_pos, len(op_lst), force_inline=force_inline)


def generate_merge_space(op_lst, down_graph, force_merge=False):
    merge_ops = list(range(len(op_lst)))
    return MergeSpce(merge_ops, len(op_lst), force_merge=force_merge)


def generate_fuse_space(loops, groups):
    return FuseSpace(groups, loops)


def generate_split_space(extent, nparts, allow_non_divisible='off'):
    return SplitSpace(nparts, extent, allow_non_divisible=allow_non_divisible)


def generate_reorder_space(num_spatial_axis):
    return ReorderSpace(num_spatial_axis)


def generate_unroll_space(explicit=False):
    return UnrollSpace([0, 1, 512, 1500], explicit=explicit)


def generate_space_intra_op(op, down_graph, slevel=4, rlevel=3, groups=3, split_policy="off", 
                            unroll_policy="off", fuse_policy="fuse_spatial", reorder_policy="last"):
    spatial_axis_names = [x.var.name for x in op.axis]
    spatial_axis_extents = [x.dom.extent.value for x in op.axis]
    reduced_axis_names = [x.var.name for x in op.reduce_axis]
    reduced_axis_extents = [x.dom.extent.value for x in op.reduce_axis]

    ##############################################################
    # generate space: 
    schedule_space = Space()

    # - fuse space
    if fuse_policy == "fuse_spatial":
        fuse_space = generate_fuse_space(spatial_axis_names, groups)
        schedule_space.add_subspace("fuse_spatial", fuse_space, "fuse")

    # - split space
    for i, (name, extent) in enumerate(zip(spatial_axis_names, spatial_axis_extents)):
        split_space = generate_split_space(extent, slevel, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "spatial")
    for i, (name, extent) in enumerate(zip(reduced_axis_names, reduced_axis_extents)):
        split_space = generate_split_space(extent, rlevel, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "reduce")

    # - reorder space
    if reorder_policy == "last":
        reorder_space = generate_reorder_space(groups)
        schedule_space.add_subspace("reorder", reorder_space, "reorder")

    # -unroll space
    unroll_space = generate_unroll_space(explicit=(unroll_policy == "explicit"))
    schedule_space.add_subspace("unroll", unroll_space, "unroll")
    
    # - other special spaces can be added   

    return schedule_space


def generate_space_inter_op(op_lst, down_graph, force_inline=False, force_merge=False, special_space=None):

    ##############################################################
    # generate space:
    schedule_space = Space()
    # - inline space
    inline_space = generate_inline_space(op_lst, down_graph, force_inline=force_inline)
    schedule_space.add_subspace("inline", inline_space, "inline")
    # - merge space
    # merge_space = generate_merge_space(op_lst, down_graph, force_merge=force_merge)
    # schedule_space.add_subspace("merge", merge_space, "merge")
    
    # - other special spaces can be added   
    special_space = {} if special_space is None else special_space
    for key, sspace in special_space.items():
        schedule_space.add_subspace(key, sspace, "special")

    return schedule_space