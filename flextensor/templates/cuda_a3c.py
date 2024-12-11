import math
from tvm import te
from flextensor.utils import assert_print
from functools import reduce

__all__ = [
    'cuda_schedule_split_fuse',
    'cuda_schedule_fuse_split',
    'cuda_schedule_split_reorder_fuse',
]


def cuda_schedule_split_fuse(config, s, op, op_state):
    # assert_print(op in s)

    # always cache write here
    # if op.num_outputs > 1:
    #     raise RuntimeWarning("Too many outputs in one operation!")
    write_cache = s.cache_write(op.output(0), "local")

    # always cache read here
    read_cache_share_lst = []
    read_cache_local_lst = []
    for t in op.input_tensors:
        share = s.cache_read(t, "shared", [write_cache])
        read_cache_share_lst.append(share)
        local = s.cache_read(share, "local", [write_cache])
        read_cache_local_lst.append(local)

    # spatial split
    spatial_axes = s[op].op.axis
    splited_spatial_axes = []
    if "spatial" in config and len(config["spatial"]) > 0:
        # to align each axis
        assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
        for axis, nparts in zip(spatial_axes, config["spatial"]):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[op].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_spatial_axes.append(tmp_buffer)
    else:
        for axis in spatial_axes:
            splited_spatial_axes.append([axis])
    assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")  # must be non-empty

    # always reorder and fuse here
    spatial_fuse_lsts = []
    spatial_fuse_extents = []
    reorder_lst = []
    fused_spatial_axes = []
    for count in range(len(splited_spatial_axes[0])):
        tmp_buffer = [x[count] for x in splited_spatial_axes]
        tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
        spatial_fuse_lsts.append(tmp_buffer)
        spatial_fuse_extents.append(tmp_extent)
        reorder_lst.extend(tmp_buffer)
    s[op].reorder(*reorder_lst)
    for fuse_lst in spatial_fuse_lsts:
        fused = s[op].fuse(*fuse_lst)
        fused_spatial_axes.append(fused)
    kernel_scope = fused_spatial_axes[0]

    # always bind here
    length = len(fused_spatial_axes)
    thread_extents = 1
    assert_print(length > 1, "fused axes length <= 1")
    if 2 <= length <= 3:
        s[op].bind(fused_spatial_axes[0], te.thread_axis("blockIdx.x"))
        s[op].bind(fused_spatial_axes[1], te.thread_axis("threadIdx.x"))
        thread_pos = fused_spatial_axes[1]
        thread_extents = spatial_fuse_extents[1]
    else:
        s[op].bind(fused_spatial_axes[0], te.thread_axis("blockIdx.x"))
        s[op].bind(fused_spatial_axes[1], te.thread_axis("vthread"))
        s[op].bind(fused_spatial_axes[2], te.thread_axis("threadIdx.x"))
        thread_pos = fused_spatial_axes[2]
        thread_extents = spatial_fuse_extents[2]

    # always compute at here
    s[write_cache].compute_at(s[op], thread_pos)

    # reduce_split
    reduced_axes = s[write_cache].op.reduce_axis
    splited_reduced_axes = []
    if "reduce" in config and len(config["reduce"]) > 0:
        # to align each axis
        assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        for axis, nparts in zip(reduced_axes, config["reduce"]):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_reduced_axes.append(tmp_buffer)
    else:
        for axis in reduced_axes:
            splited_reduced_axes.append([axis])
    share_pos = None
    local_pos = None
    # if has reduce axes
    if len(splited_reduced_axes) > 0:
        # always reorder here
        reduced_nonfuse_lsts = []
        reorder_lst = []
        length = len(splited_reduced_axes[0])

        for count in range(length):
            tmp_buffer = [x[count] for x in splited_reduced_axes]
            reduced_nonfuse_lsts.append(tmp_buffer)
            reorder_lst.extend(tmp_buffer)
        # change the order of reduce axes and spatial axes
        reorder_lst.extend(s[write_cache].op.axis)
        s[write_cache].reorder(*reorder_lst)

        if length == 1:
            share_pos = reduced_nonfuse_lsts[-1][0]
        else:
            share_pos = reduced_nonfuse_lsts[-2][0]
            local_pos = reduced_nonfuse_lsts[-1][-1]

    # always cache read here
    if share_pos is not None:
        for share in read_cache_share_lst:
            s[share].compute_at(s[write_cache], share_pos)
    else:
        for share in read_cache_share_lst:
            s[share].compute_inline()
    if local_pos is not None:
        for local in read_cache_local_lst:
            s[local].compute_at(s[write_cache], local_pos)
    else:
        for local in read_cache_local_lst:
            s[local].compute_inline()

    # always cooperative fetching
    if share_pos is not None:
        for share in read_cache_share_lst:
            fuse_lst = s[share].op.axis
            fused = s[share].fuse(*fuse_lst)
            outer, inner = s[share].split(fused, nparts=thread_extents)
            s[share].bind(outer, te.thread_axis("threadIdx.x"))

    # unroll
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)


def cuda_schedule_fuse_split(config, s, op, op_state):
    # assert_print(op in s)

    # always cache write here
    # if op.num_outputs > 1:
    #     raise RuntimeWarning("Too many outputs in one operation!")
    write_cache = s.cache_write(op.output(0), "local")

    # always cache read here
    read_cache_share_lst = []
    # read_cache_local_lst = []
    for t in op.input_tensors:
        share = s.cache_read(t, "shared", [write_cache])
        read_cache_share_lst.append(share)
        # local = s.cache_read(share, "local", [write_cache])
        # read_cache_local_lst.append(local)

    # spatial fuse
    spatial_axes = s[op].op.axis
    fused_spatial_axes = []
    if "fuse" in config and len(config["fuse"]) > 0:
        # fuse redundant axes
        beg = 0
        for end in config["fuse"][0]:
            fuse_lst = spatial_axes[beg:end]
            beg = end
            if len(fuse_lst) > 0:
                fused = s[op].fuse(*fuse_lst)
                fused_spatial_axes.append(fused)
    else:
        fused_spatial_axes = spatial_axes

    # spatial split
    split_factor_lst = []
    splited_spatial_axes = []
    if "spatial" in config and len(config["spatial"]) > 0:
        # to align each axis
        assert len(config["spatial"]) == len(spatial_axes), "align failed"
        # compute split factors
        if "fuse" in config and len(config["fuse"]) > 0:
            beg = 0
            for end in config["fuse"][0]:
                tmp_lst = [1] * len(config["spatial"][0])
                for i in range(beg, end):
                    for j in range(len(config["spatial"][i])):
                        tmp_lst[j] *= config["spatial"][i][j]
                if beg < end:
                    split_factor_lst.append(tmp_lst)
                beg = end
        else:
            split_factor_lst = config["spatial"]
        assert len(fused_spatial_axes) == len(split_factor_lst), "align failed"
        for axis, nparts in zip(fused_spatial_axes, split_factor_lst):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[op].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_spatial_axes.append(tmp_buffer)
    else:
        for axis in fused_spatial_axes:
            splited_spatial_axes.append([axis])
    assert len(splited_spatial_axes) > 0, "empty spatial axes"  # must be non-empty

    # always reorder here
    reorder_lst = []
    for count in range(len(splited_spatial_axes[0])):
        tmp_buffer = [x[count] for x in splited_spatial_axes]
        reorder_lst.extend(tmp_buffer)
    s[op].reorder(*reorder_lst)

    # fix kernel scope
    kernel_scope = reorder_lst[0]

    # always bind here
    # - prepare thread axis
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    bz = te.thread_axis("blockIdx.z")
    vx = te.thread_axis("vthread")
    vy = te.thread_axis("vthread")
    vz = te.thread_axis("vthread")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    tz = te.thread_axis("threadIdx.z")

    blocks = [bz, by, bx]
    threads = [tz, ty, tx]
    vthreads = [vz, vy, vx]

    block_extents = [-1, -1, -1]  # z, y, x
    virtual_extents = [-1, -1, -1]
    thread_extents = [-1, -1, -1]

    length = len(splited_spatial_axes)
    assert length >= 1
    # - bind
    count = min(length, len(blocks)) - 1
    while count >= 0:
        parts = len(splited_spatial_axes[count])
        assert parts > 0
        if parts == 1:
            s[op].bind(splited_spatial_axes[count][0], blocks[count])
            block_extents[count] = split_factor_lst[count][0]
        elif parts == 2:
            s[op].bind(splited_spatial_axes[count][0], blocks[count])
            block_extents[count] = split_factor_lst[count][0]
            s[op].bind(splited_spatial_axes[count][1], threads[count])
            thread_extents[count] = split_factor_lst[count][1]
        else:
            s[op].bind(splited_spatial_axes[count][0], blocks[count])
            block_extents[count] = split_factor_lst[count][0]
            s[op].bind(splited_spatial_axes[count][1], vthreads[count])
            virtual_extents[count] = split_factor_lst[count][1]
            s[op].bind(splited_spatial_axes[count][2], threads[count])
            thread_extents[count] = split_factor_lst[count][2]
        count -= 1
    # - compute at pos
    count = min(length, len(blocks)) - 1
    parts = len(splited_spatial_axes[count])
    thread_pos = splited_spatial_axes[count][min(parts - 1, 2)]

    # always compute at here
    s[write_cache].compute_at(s[op], thread_pos)

    # reduce_split
    reduced_axes = s[write_cache].op.reduce_axis
    splited_reduced_axes = []
    if "reduce" in config and len(config["reduce"]) > 0:
        # to align each axis
        assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        for axis, nparts in zip(reduced_axes, config["reduce"]):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_reduced_axes.append(tmp_buffer)
    else:
        for axis in reduced_axes:
            splited_reduced_axes.append([axis])
    share_pos = None
    # local_pos = None
    # if has reduce axes
    if len(splited_reduced_axes) > 0:
        # always reorder here
        reduced_nonfuse_lsts = []
        reorder_lst = []
        length = len(splited_reduced_axes[0])
        # leave the last part
        for count in range(length - 1):
            tmp_buffer = [x[count] for x in splited_reduced_axes]
            reduced_nonfuse_lsts.append(tmp_buffer)
            reorder_lst.extend(tmp_buffer)
        # the last part
        last_part = [x[length - 1] for x in splited_reduced_axes]
        spatial_remainder = s[write_cache].op.axis
        # change the order of reduce axes and spatial axes
        if "reorder" in config and len(config["reorder"]) > 0:
            pos = config["reorder"][0][0]
            assert pos < len(spatial_remainder)
            tmp_buffer = []
            count = len(spatial_remainder) - 1
            while count > pos:
                tmp_buffer.append(spatial_remainder[count])
                count -= 1
            p = pos
            q = len(last_part) - 1
            while p >= 0 and q >= 0:
                tmp_buffer.append(spatial_remainder[p])
                tmp_buffer.append(last_part[q])
                p -= 1
                q -= 1
            while p >= 0:
                tmp_buffer.append(spatial_remainder[p])
                p -= 1
            while q >= 0:
                tmp_buffer.append(last_part[q])
                q -= 1
            tmp_buffer = list(reversed(tmp_buffer))
            reorder_lst.extend(tmp_buffer)
        else:
            reorder_lst.extend(last_part)
            reorder_lst.extend(spatial_remainder)
        s[write_cache].reorder(*reorder_lst)
        # decide where to compute at
        if length == 1:
            share_pos = last_part[-1]
        else:
            mid = math.ceil(length / 2.0) - 1
            share_pos = reduced_nonfuse_lsts[mid][-1]
            # local_pos = last_part[-1]

    # always cache read here
    if share_pos is not None:
        for share in read_cache_share_lst:
            s[share].compute_at(s[write_cache], share_pos)
    else:
        for share in read_cache_share_lst:
            s[share].compute_inline()
    # if local_pos is not None:
    #     for local in read_cache_local_lst:
    #         s[local].compute_at(s[write_cache], local_pos)
    # else:
    #     for local in read_cache_local_lst:
    #         s[local].compute_inline()

    # always cooperative fetching
    if share_pos is not None:
        for share in read_cache_share_lst:
            fuse_lst = s[share].op.axis
            fused = s[share].fuse(*fuse_lst)
            count = 2
            cur = 1
            limit = 1024
            while count >= 0:
                factor = thread_extents[count]
                if factor < 0:
                    defined = False
                    factor = 16
                else:
                    defined = True
                cur *= factor
                if not defined and cur > limit:
                    break
                fused, inner = s[share].split(fused, factor=factor)
                s[share].bind(inner, threads[count])
                count -= 1

    # unroll
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)


def cuda_schedule_split_reorder_fuse(config, s, op, op_state):
    # assert_print(op in s)

    loop_lst = []
    loop_idx = []

    # always cache write here
    # if op.num_outputs > 1:
    #     raise RuntimeWarning("Too many outputs in one operation!")
    write_cache = s.cache_write(op.output(0), "local")
    # always cache read here
    read_cache_share_lst = []
    # read_cache_local_lst = []
    for t in op.input_tensors:
        share = s.cache_read(t, "shared", [write_cache])
        read_cache_share_lst.append(share)
        # local = s.cache_read(share, "local", [write_cache])
        # read_cache_local_lst.append(local)

    # spatial split
    spatial_axes = [axis for axis in s[op].op.axis]
    assert len(spatial_axes) > 0, "empty spatial axes"  # must be non-empty
    n = spatial_axes[0]
    kernel_scope, n = s[op].split(n, nparts=1)
    spatial_axes[0] = n
    splited_spatial_axes = []
    splited_spatial_extents = []
    if "spatial" in config and len(config["spatial"]) > 0:
        # to align each axis
        assert len(config["spatial"]) == len(spatial_axes), "align failed"
        for axis, nparts in zip(spatial_axes, config["spatial"]):
            tmp_buffer = []
            tmp_extents = []
            for count in range(len(nparts) - 1):
                outer, axis = s[op].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
                tmp_extents.append(nparts[count])
            tmp_buffer.append(axis)
            tmp_extents.append(nparts[-1])
            splited_spatial_axes.append(tmp_buffer)
            # splited_spatial_extents.append(tmp_extents)
            for axis in s[op].op.axis:
                if axis.dom is not None and axis.dom.extent is not None:
                    splited_spatial_extents.append([axis.dom.extent.value])
                else:
                    # axis.dom이 None인 경우 처리 (skip하거나 기본값 사용)
                    splited_spatial_extents.append([1]) # or continue
    else:
        for axis in spatial_axes:
            splited_spatial_axes.append([axis])
            # splited_spatial_extents.append([axis.dom.extent.value])
            for axis in s[op].op.axis:
                if axis.dom is not None and axis.dom.extent is not None:
                    splited_spatial_extents.append([axis.dom.extent.value])
                else:
                    # axis.dom이 None인 경우 처리 (skip하거나 기본값 사용)
                    splited_spatial_extents.append([1]) # or continue

    # always reorder here
    reorder_lst = []
    reorder_parts = []
    reorder_part_extents = []
    for count in range(len(splited_spatial_axes[0])):
        tmp_buffer = [x[count] for x in splited_spatial_axes]
        tmp_extents = [x[count] for x in splited_spatial_extents]
        reorder_lst.extend(tmp_buffer)
        reorder_parts.append(tmp_buffer)
        reorder_part_extents.append(tmp_extents)
    s[op].reorder(*reorder_lst)
    # handle fuse request
    fused_parts = []
    fused_part_extents = []
    fused_part_idx = []
    if "fuse" in config and len(config["fuse"]) > 0:
        base_id = 0
        for part, extents in zip(reorder_parts, reorder_part_extents):
            tmp_part = []
            tmp_extents = []
            tmp_idx = []
            idx = 0
            beg = 0
            for end in config["fuse"][0]:
                if end - beg > 1:
                    fuse_lst = part[beg:end]
                    fused = s[op].fuse(*fuse_lst)
                    tmp_part.append(fused)
                    extent = reduce(lambda x, y: x * y, extents[beg:end], 1)
                    tmp_idx.extend([idx] * (end - beg))
                    idx += 1
                    tmp_extents.append(extent)
                elif end - beg == 1:
                    tmp_part.append(part[beg])
                    tmp_extents.append(extents[beg])
                    tmp_idx.append(idx)
                    idx += 1
                beg = end
            fused_parts.append(tmp_part)
            fused_part_extents.append(tmp_extents)
            fused_part_idx.append(tmp_idx)

            loop_lst.extend(tmp_part)
            loop_idx.extend([x + base_id for x in tmp_idx])
            base_id += len(tmp_part)
    else:
        fused_parts = reorder_parts
        fused_part_extents = reorder_part_extents
        fused_part_idx = [list(range(len(x))) for x in reorder_parts]

        loop_lst = reorder_lst
        loop_idx = list(range(len(reorder_lst)))
    # record op state
    op_state.loop_lst = loop_lst
    op_state.loop_idx = loop_idx

    # always bind here
    # - prepare thread axis
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    bz = te.thread_axis("blockIdx.z")
    vx = te.thread_axis("vthread")
    vy = te.thread_axis("vthread")
    vz = te.thread_axis("vthread")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    tz = te.thread_axis("threadIdx.z")

    blocks = [bz, by, bx]
    threads = [tz, ty, tx]
    vthreads = [vz, vy, vx]

    block_extents = [-1, -1, -1]  # z, y, x
    virtual_extents = [-1, -1, -1]
    thread_extents = [-1, -1, -1]

    bind_option = [None, None, None]
    bind_candidate = [blocks, vthreads, threads]
    candiate_extents = [block_extents, virtual_extents, thread_extents]

    # - bind
    num_parts = len(fused_parts)
    if num_parts == 1:
        bind_option[0] = (fused_parts[0], fused_part_extents[0])
        local_pos = fused_parts[0][:len(bind_candidate[0])][-1]
    elif num_parts == 2:
        bind_option[0] = (fused_parts[0], fused_part_extents[0])
        bind_option[2] = (fused_parts[1], fused_part_extents[1])
        local_pos = fused_parts[1][:len(bind_candidate[2])][-1]
    else:
        bind_option[0] = (fused_parts[0], fused_part_extents[0])
        bind_option[1] = (fused_parts[1], fused_part_extents[1])
        bind_option[2] = (fused_parts[2], fused_part_extents[2])
        local_pos = fused_parts[2][:len(bind_candidate[2])][-1]
    for option, candidate, extents in zip(bind_option, bind_candidate, candiate_extents):
        if option is not None:
            for i, axis in enumerate(option[0][:len(candidate)]):
                s[op].bind(axis, candidate[i])
                extents[i] = option[1][i]
    # compute at
    if "local_pos" in config and len(config["local_pos"]) > 0:
        local_at_part = config["local_pos"][0][0]
        local_at_idx = config["local_pos"][0][1]
        # index changed because of fusion
        cur_idx = fused_part_idx[local_at_part][local_at_idx]
        local_pos = fused_parts[local_at_part][cur_idx]

    # always compute at here
    s[write_cache].compute_at(s[op], local_pos)

    # reduce_split
    reduced_axes = s[write_cache].op.reduce_axis
    splited_reduced_axes = []
    if "reduce" in config and len(config["reduce"]) > 0:
        # to align each axis
        assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        for axis, nparts in zip(reduced_axes, config["reduce"]):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_reduced_axes.append(tmp_buffer)
    else:
        for axis in reduced_axes:
            splited_reduced_axes.append([axis])
    share_pos = None
    # local_pos = None
    # if has reduce axes
    if len(splited_reduced_axes) > 0:
        # always reorder here
        reduced_nonfuse_lsts = []
        reorder_lst = []
        length = len(splited_reduced_axes[0])
        # leave the last part
        for count in range(length - 1):
            tmp_buffer = [x[count] for x in splited_reduced_axes]
            reduced_nonfuse_lsts.append(tmp_buffer)
            reorder_lst.extend(tmp_buffer)
        # the last part
        last_part = [x[length - 1] for x in splited_reduced_axes]
        spatial_remainder = s[write_cache].op.axis
        # change the order of reduce axes and spatial axes
        if "reorder" in config and len(config["reorder"]) > 0:
            pos = config["reorder"][0][0]
            assert pos < len(spatial_remainder)
            tmp_buffer = []
            count = len(spatial_remainder) - 1
            while count > pos:
                tmp_buffer.append(spatial_remainder[count])
                count -= 1
            p = pos
            q = len(last_part) - 1
            while p >= 0 and q >= 0:
                tmp_buffer.append(spatial_remainder[p])
                tmp_buffer.append(last_part[q])
                p -= 1
                q -= 1
            while p >= 0:
                tmp_buffer.append(spatial_remainder[p])
                p -= 1
            while q >= 0:
                tmp_buffer.append(last_part[q])
                q -= 1
            tmp_buffer = list(reversed(tmp_buffer))
            reorder_lst.extend(tmp_buffer)
        else:
            reorder_lst.extend(last_part)
            reorder_lst.extend(spatial_remainder)
        s[write_cache].reorder(*reorder_lst)
        # decide where to compute at
        if "share_pos" in config and len(config["share_pos"]) > 0:
            share_at = config["share_pos"][0][0]
            share_idx = config["share_pos"][0][1]
            reduced_nonfuse_lsts.append(last_part)
            share_pos = reduced_nonfuse_lsts[share_at][share_idx]
        else:
            if length == 1:
                share_pos = last_part[-1]
            else:
                mid = math.ceil(length / 2.0) - 1
                share_pos = reduced_nonfuse_lsts[mid][-1]
                # local_pos = last_part[-1]

    # always cache read here
    if share_pos is not None:
        for share in read_cache_share_lst:
            s[share].compute_at(s[write_cache], share_pos)
    else:
        for share in read_cache_share_lst:
            s[share].compute_inline()
    # if local_pos is not None:
    #     for local in read_cache_local_lst:
    #         s[local].compute_at(s[write_cache], local_pos)
    # else:
    #     for local in read_cache_local_lst:
    #         s[local].compute_inline()

    # always cooperative fetching
    if share_pos is not None:
        for share in read_cache_share_lst:
            fuse_lst = s[share].op.axis
            fused = s[share].fuse(*fuse_lst)
            count = 2
            cur = 1
            limit = 1024
            while count >= 0:
                factor = thread_extents[count]
                if factor < 0:
                    defined = False
                    factor = 16
                else:
                    defined = True
                cur *= factor
                if not defined and cur > limit:
                    break
                fused, inner = s[share].split(fused, factor=factor)
                s[share].bind(inner, threads[count])
                count -= 1

    # unroll
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)
