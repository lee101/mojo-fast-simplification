"""Quadric-error triangle-mesh decimation over caller-owned buffers."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime JPtr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime HUGE = 1.7976931348623157e308


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def jp(address: Int) -> JPtr:
    return JPtr(unsafe_from_address=address)


def up(address: Int) -> UPtr:
    return UPtr(unsafe_from_address=address)


struct V3(Copyable, Movable, ImplicitlyCopyable, ImplicitlyDeletable):
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: V3) -> V3:
        return V3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: V3) -> V3:
        return V3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: Float64) -> V3:
        return V3(self.x * scale, self.y * scale, self.z * scale)

    def dot(self, other: V3) -> Float64:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: V3) -> V3:
        return V3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def squared_norm(self) -> Float64:
        return self.dot(self)


@always_inline
def load_v(points: FPtr, i: Int) -> V3:
    return V3(points[3 * i], points[3 * i + 1], points[3 * i + 2])


@always_inline
def store_v(points: FPtr, i: Int, value: V3):
    points[3 * i] = value.x
    points[3 * i + 1] = value.y
    points[3 * i + 2] = value.z


@always_inline
def add_plane(q: FPtr, vertex: Int, a: Float64, b: Float64, c: Float64, d: Float64):
    var base = 10 * vertex
    q[base] += a * a
    q[base + 1] += a * b
    q[base + 2] += a * c
    q[base + 3] += a * d
    q[base + 4] += b * b
    q[base + 5] += b * c
    q[base + 6] += b * d
    q[base + 7] += c * c
    q[base + 8] += c * d
    q[base + 9] += d * d


@always_inline
def determinant(
    a11: Float64,
    a12: Float64,
    a13: Float64,
    a21: Float64,
    a22: Float64,
    a23: Float64,
    a31: Float64,
    a32: Float64,
    a33: Float64,
) -> Float64:
    return (
        a11 * a22 * a33
        + a13 * a21 * a32
        + a12 * a23 * a31
        - a13 * a22 * a31
        - a11 * a23 * a32
        - a12 * a21 * a33
    )


@always_inline
def qvalue(q: FPtr, u: Int, v: Int, p: V3) -> Float64:
    var q0 = q[10 * u] + q[10 * v]
    var q1 = q[10 * u + 1] + q[10 * v + 1]
    var q2 = q[10 * u + 2] + q[10 * v + 2]
    var q3 = q[10 * u + 3] + q[10 * v + 3]
    var q4 = q[10 * u + 4] + q[10 * v + 4]
    var q5 = q[10 * u + 5] + q[10 * v + 5]
    var q6 = q[10 * u + 6] + q[10 * v + 6]
    var q7 = q[10 * u + 7] + q[10 * v + 7]
    var q8 = q[10 * u + 8] + q[10 * v + 8]
    var q9 = q[10 * u + 9] + q[10 * v + 9]
    return (
        q0 * p.x * p.x
        + 2.0 * q1 * p.x * p.y
        + 2.0 * q2 * p.x * p.z
        + 2.0 * q3 * p.x
        + q4 * p.y * p.y
        + 2.0 * q5 * p.y * p.z
        + 2.0 * q6 * p.y
        + q7 * p.z * p.z
        + 2.0 * q8 * p.z
        + q9
    )


@always_inline
def candidate(points: FPtr, q: FPtr, border: IPtr, u: Int, v: Int) -> Tuple[V3, Float64, Float64]:
    var p0 = load_v(points, u)
    var p1 = load_v(points, v)
    var midpoint = (p0 + p1) * 0.5
    var position = midpoint
    var error = 0.0
    if border[u] == 0 or border[v] == 0:
        var q0 = q[10 * u] + q[10 * v]
        var q1 = q[10 * u + 1] + q[10 * v + 1]
        var q2 = q[10 * u + 2] + q[10 * v + 2]
        var q3 = q[10 * u + 3] + q[10 * v + 3]
        var q4 = q[10 * u + 4] + q[10 * v + 4]
        var q5 = q[10 * u + 5] + q[10 * v + 5]
        var q6 = q[10 * u + 6] + q[10 * v + 6]
        var q7 = q[10 * u + 7] + q[10 * v + 7]
        var q8 = q[10 * u + 8] + q[10 * v + 8]
        var det = determinant(q0, q1, q2, q1, q4, q5, q2, q5, q7)
        if abs(det) > 1.0e-15:
            position = V3(
                -determinant(q1, q2, q3, q4, q5, q6, q5, q7, q8) / det,
                determinant(q0, q2, q3, q1, q5, q6, q2, q7, q8) / det,
                -determinant(q0, q1, q3, q1, q4, q6, q2, q5, q8) / det,
            )
            error = qvalue(q, u, v, position)
        else:
            var error0 = qvalue(q, u, v, p0)
            var error1 = qvalue(q, u, v, p1)
            var error_mid = qvalue(q, u, v, midpoint)
            error = error_mid
            if error0 <= error1 and error0 <= error_mid:
                position = p0
                error = error0
            elif error1 <= error_mid:
                position = p1
                error = error1
    else:
        var error0 = qvalue(q, u, v, p0)
        var error1 = qvalue(q, u, v, p1)
        var error_mid = qvalue(q, u, v, midpoint)
        error = error_mid
        if error0 <= error1 and error0 <= error_mid:
            position = p0
            error = error0
        elif error1 <= error_mid:
            position = p1
            error = error1
    var edge = p1 - p0
    var denom = edge.squared_norm()
    var blend = 0.5
    if denom > 0.0:
        blend = (position - p0).dot(edge) / denom
        if blend < 0.0:
            blend = 0.0
        elif blend > 1.0:
            blend = 1.0
    return (position, max(error, 0.0), blend)


@always_inline
def total_error(
    points: FPtr,
    attrs: FPtr,
    weights: FPtr,
    q: FPtr,
    border: IPtr,
    u: Int,
    v: Int,
    attr_dim: Int,
) -> Float64:
    var _, error, _ = candidate(points, q, border, u, v)
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    while k + W <= attr_dim:
        var delta = attrs.load[width=W](u * attr_dim + k) - attrs.load[width=W](
            v * attr_dim + k
        )
        error += (
            0.5
            * (
                weights.load[width=W](k)
                * delta
                * delta
            ).reduce_add()
        )
        k += W
    while k < attr_dim:
        var delta = attrs[u * attr_dim + k] - attrs[v * attr_dim + k]
        error += 0.5 * weights[k] * delta * delta
        k += 1
    return error


@always_inline
def heap_less(errors: FPtr, edges: UPtr, a: Int, b: Int) -> Bool:
    if errors[a] != errors[b]:
        return errors[a] < errors[b]
    return edges[a] < edges[b]


@always_inline
def heap_swap(
    errors: FPtr, edges: UPtr, version_pairs: UPtr, a: Int, b: Int
):
    var te = errors[a]
    errors[a] = errors[b]
    errors[b] = te
    var ti = edges[a]
    edges[a] = edges[b]
    edges[b] = ti
    ti = version_pairs[a]
    version_pairs[a] = version_pairs[b]
    version_pairs[b] = ti


def heap_push(
    errors: FPtr,
    edges: UPtr,
    version_pairs: UPtr,
    size: Int,
    capacity: Int,
    error: Float64,
    u: Int,
    v: Int,
    version_u: Int,
    version_v: Int,
) -> Int:
    if size >= capacity or u == v:
        return size
    var new_size = size
    var pos = new_size
    errors[pos] = error
    edges[pos] = (UInt64(u) << 32) | UInt64(v)
    version_pairs[pos] = (UInt64(version_u) << 32) | UInt64(version_v)
    new_size += 1
    while pos > 0:
        var parent = (pos - 1) // 2
        if not heap_less(errors, edges, pos, parent):
            break
        heap_swap(errors, edges, version_pairs, pos, parent)
        pos = parent
    return new_size


def heap_pop(
    errors: FPtr, edges: UPtr, version_pairs: UPtr, size: Int
) -> Tuple[Int, Int, Int, Int, Int, Float64]:
    var edge = edges[0]
    var versions = version_pairs[0]
    var u = Int(edge >> 32)
    var v = Int(edge & UInt64(0xFFFFFFFF))
    var version_u = Int(versions >> 32)
    var version_v = Int(versions & UInt64(0xFFFFFFFF))
    var error = errors[0]
    var new_size = size - 1
    if new_size > 0:
        errors[0] = errors[new_size]
        edges[0] = edges[new_size]
        version_pairs[0] = version_pairs[new_size]
        var pos = 0
        while True:
            var left = 2 * pos + 1
            if left >= new_size:
                break
            var child = left
            var right = left + 1
            if right < new_size and heap_less(errors, edges, right, left):
                child = right
            if not heap_less(errors, edges, child, pos):
                break
            heap_swap(errors, edges, version_pairs, child, pos)
            pos = child
    return (new_size, u, v, version_u, version_v, error)


@always_inline
def face_contains(faces: JPtr, face: Int, vertex: Int) -> Bool:
    return (
        Int(faces[3 * face]) == vertex
        or Int(faces[3 * face + 1]) == vertex
        or Int(faces[3 * face + 2]) == vertex
    )


def edge_face_count(
    faces: JPtr, face_alive: IPtr, head: IPtr, next_ref: IPtr, u: Int, v: Int
) -> Int:
    var count = 0
    var node = Int(head[u])
    while node >= 0:
        var face = node // 3
        if face_alive[face] != 0 and face_contains(faces, face, v):
            count += 1
        node = Int(next_ref[node])
    return count


def collapse_valid(
    points: FPtr,
    faces: JPtr,
    face_alive: IPtr,
    border: IPtr,
    head: IPtr,
    next_ref: IPtr,
    marks: IPtr,
    u: Int,
    v: Int,
    position: V3,
    stamp: Int,
) -> Bool:
    var edge_faces = edge_face_count(faces, face_alive, head, next_ref, u, v)
    if edge_faces < 1 or edge_faces > 2 or border[u] != border[v]:
        return False
    if border[u] != 0 and edge_faces != 1:
        return False

    var node = Int(head[u])
    while node >= 0:
        var face = node // 3
        if face_alive[face] != 0:
            for j in range(3):
                var w = Int(faces[3 * face + j])
                if w != u:
                    marks[w] = Int64(stamp)
        node = Int(next_ref[node])
    var common = 0
    node = Int(head[v])
    while node >= 0:
        var face = node // 3
        if face_alive[face] != 0:
            for j in range(3):
                var w = Int(faces[3 * face + j])
                if w != v and Int(marks[w]) == stamp:
                    common += 1
                    marks[w] = Int64(-stamp)
        node = Int(next_ref[node])
    if common != edge_faces:
        return False

    node = Int(head[u])
    while node >= 0:
        var face = node // 3
        if face_alive[face] != 0 and not face_contains(faces, face, v):
            if face_would_flip(points, faces, face, u, position):
                return False
        node = Int(next_ref[node])
    node = Int(head[v])
    while node >= 0:
        var face = node // 3
        if face_alive[face] != 0 and not face_contains(faces, face, u):
            if face_would_flip(points, faces, face, v, position):
                return False
        node = Int(next_ref[node])
    return True


def face_would_flip(
    points: FPtr, faces: JPtr, face: Int, replaced: Int, position: V3
) -> Bool:
    var a = load_v(points, Int(faces[3 * face]))
    var b = load_v(points, Int(faces[3 * face + 1]))
    var c = load_v(points, Int(faces[3 * face + 2]))
    var old_normal = (b - a).cross(c - a)
    if Int(faces[3 * face]) == replaced:
        a = position
    elif Int(faces[3 * face + 1]) == replaced:
        b = position
    else:
        c = position
    var new_normal = (b - a).cross(c - a)
    var old_sq = old_normal.squared_norm()
    var new_sq = new_normal.squared_norm()
    if old_sq <= 1.0e-30 or new_sq <= 1.0e-30:
        return True
    var alignment = new_normal.dot(old_normal)
    return alignment < 0.0 or alignment * alignment < 0.04 * new_sq * old_sq


def rebuild_heap(
    points: FPtr,
    faces: JPtr,
    attrs: FPtr,
    weights: FPtr,
    q: FPtr,
    face_alive: IPtr,
    vertex_alive: IPtr,
    border: IPtr,
    head: IPtr,
    next_ref: IPtr,
    versions: IPtr,
    marks: IPtr,
    heap_errors: FPtr,
    heap_edges: UPtr,
    heap_versions: UPtr,
    vertex_count: Int,
    attr_dim: Int,
    capacity: Int,
) -> Int:
    var size = 0
    for u in range(vertex_count):
        if vertex_alive[u] == 0:
            continue
        var marker = -u - 1
        var node = Int(head[u])
        while node >= 0:
            var face = node // 3
            if face_alive[face] != 0:
                var corner = node % 3
                for offset in range(1, 3):
                    var v = Int(faces[3 * face + (corner + offset) % 3])
                    if (
                        v > u
                        and vertex_alive[v] != 0
                        and Int(marks[v]) != marker
                    ):
                        marks[v] = Int64(marker)
                        var error = total_error(
                            points, attrs, weights, q, border, u, v, attr_dim
                        )
                        if size < capacity:
                            heap_errors[size] = error
                            heap_edges[size] = (UInt64(u) << 32) | UInt64(v)
                            heap_versions[size] = (
                                (UInt64(versions[u]) << 32)
                                | UInt64(versions[v])
                            )
                            size += 1
            node = Int(next_ref[node])

    var parent = size // 2
    while parent > 0:
        parent -= 1
        var pos = parent
        while True:
            var left = 2 * pos + 1
            if left >= size:
                break
            var child = left
            var right = left + 1
            if right < size and heap_less(
                heap_errors, heap_edges, right, left
            ):
                child = right
            if not heap_less(heap_errors, heap_edges, child, pos):
                break
            heap_swap(
                heap_errors,
                heap_edges,
                heap_versions,
                child,
                pos,
            )
            pos = child
    return size


def simplify_core(
    points: FPtr,
    faces: JPtr,
    attrs: FPtr,
    weights: FPtr,
    q: FPtr,
    face_alive: IPtr,
    vertex_alive: IPtr,
    border: IPtr,
    head: IPtr,
    next_ref: IPtr,
    versions: IPtr,
    marks: IPtr,
    mapping: IPtr,
    heap_errors: FPtr,
    heap_edges: UPtr,
    heap_versions: UPtr,
    collapses: JPtr,
    result: IPtr,
    vertex_count: Int,
    face_count: Int,
    attr_dim: Int,
    target_faces: Int,
    max_error: Float64,
    heap_capacity: Int,
):
    comptime W = simd_width_of[DType.float64]()
    for i in range(vertex_count):
        vertex_alive[i] = 1
        border[i] = 0
        head[i] = -1
        versions[i] = 0
        marks[i] = 0
        mapping[i] = -1
        var j = 0
        while j + W <= 10:
            q.store(10 * i + j, SIMD[DType.float64, W](0.0))
            j += W
        while j < 10:
            q[10 * i + j] = 0.0
            j += 1
    for face in range(face_count):
        face_alive[face] = 1
        for j in range(3):
            var vertex = Int(faces[3 * face + j])
            var node = 3 * face + j
            next_ref[node] = head[vertex]
            head[vertex] = Int64(node)
        var a = load_v(points, Int(faces[3 * face]))
        var b = load_v(points, Int(faces[3 * face + 1]))
        var c = load_v(points, Int(faces[3 * face + 2]))
        var normal = (b - a).cross(c - a)
        var length = sqrt(normal.squared_norm())
        if length > 1.0e-30:
            normal = normal * (1.0 / length)
            var d = -normal.dot(a)
            for j in range(3):
                add_plane(q, Int(faces[3 * face + j]), normal.x, normal.y, normal.z, d)

    for u in range(vertex_count):
        var marker = -vertex_count - u - 1
        var node = Int(head[u])
        while node >= 0:
            var face = node // 3
            var corner = node % 3
            for offset in range(1, 3):
                var v = Int(faces[3 * face + (corner + offset) % 3])
                if v > u and Int(marks[v]) != marker:
                    marks[v] = Int64(marker)
                    if (
                        edge_face_count(
                            faces, face_alive, head, next_ref, u, v
                        )
                        == 1
                    ):
                        border[u] = 1
                        border[v] = 1
            node = Int(next_ref[node])

    var heap_size = rebuild_heap(
        points,
        faces,
        attrs,
        weights,
        q,
        face_alive,
        vertex_alive,
        border,
        head,
        next_ref,
        versions,
        marks,
        heap_errors,
        heap_edges,
        heap_versions,
        vertex_count,
        attr_dim,
        heap_capacity,
    )
    var active_faces = face_count
    var collapse_count = 0
    var stamp = 1
    while active_faces > target_faces and heap_size > 0:
        var u: Int
        var v: Int
        var version_u: Int
        var version_v: Int
        var queued_error: Float64
        heap_size, u, v, version_u, version_v, queued_error = heap_pop(
            heap_errors, heap_edges, heap_versions, heap_size
        )
        if vertex_alive[u] == 0 or vertex_alive[v] == 0:
            continue
        if version_u != Int(versions[u]) or version_v != Int(versions[v]):
            continue
        if queued_error > max_error:
            break
        var position, _, blend = candidate(points, q, border, u, v)
        stamp += 1
        if not collapse_valid(
            points,
            faces,
            face_alive,
            border,
            head,
            next_ref,
            marks,
            u,
            v,
            position,
            stamp,
        ):
            continue

        store_v(points, u, position)
        var k = 0
        while k + W <= attr_dim:
            var attr_u = attrs.load[width=W](u * attr_dim + k)
            var attr_v = attrs.load[width=W](v * attr_dim + k)
            attrs.store(
                u * attr_dim + k,
                (1.0 - blend) * attr_u + blend * attr_v,
            )
            k += W
        while k < attr_dim:
            attrs[u * attr_dim + k] = (
                (1.0 - blend) * attrs[u * attr_dim + k]
                + blend * attrs[v * attr_dim + k]
            )
            k += 1
        k = 0
        while k + W <= 10:
            q.store(
                10 * u + k,
                q.load[width=W](10 * u + k) + q.load[width=W](10 * v + k),
            )
            k += W
        while k < 10:
            q[10 * u + k] += q[10 * v + k]
            k += 1
        vertex_alive[v] = 0
        versions[u] += 1
        collapses[2 * collapse_count] = Int32(u)
        collapses[2 * collapse_count + 1] = Int32(v)
        collapse_count += 1

        var node = Int(head[v])
        while node >= 0:
            var saved_next = Int(next_ref[node])
            var face = node // 3
            if face_alive[face] != 0:
                if face_contains(faces, face, u):
                    face_alive[face] = 0
                    active_faces -= 1
                else:
                    var corner = node % 3
                    faces[3 * face + corner] = Int32(u)
                    next_ref[node] = head[u]
                    head[u] = Int64(node)
            node = saved_next

        if heap_size > 3 * face_count:
            heap_size = rebuild_heap(
                points,
                faces,
                attrs,
                weights,
                q,
                face_alive,
                vertex_alive,
                border,
                head,
                next_ref,
                versions,
                marks,
                heap_errors,
                heap_edges,
                heap_versions,
                vertex_count,
                attr_dim,
                heap_capacity,
            )
        else:
            stamp += 1
            node = Int(head[u])
            while node >= 0:
                var face = node // 3
                if face_alive[face] != 0:
                    var corner = node % 3
                    for offset in range(1, 3):
                        var w = Int(faces[3 * face + (corner + offset) % 3])
                        if Int(marks[w]) != stamp:
                            marks[w] = Int64(stamp)
                            var a = min(u, w)
                            var b = max(u, w)
                            var error = total_error(
                                points, attrs, weights, q, border, a, b, attr_dim
                            )
                            heap_size = heap_push(
                                heap_errors,
                                heap_edges,
                                heap_versions,
                                heap_size,
                                heap_capacity,
                                error,
                                a,
                                b,
                                Int(versions[a]),
                                Int(versions[b]),
                            )
                node = Int(next_ref[node])

    for face in range(face_count):
        if face_alive[face] != 0:
            for j in range(3):
                mapping[Int(faces[3 * face + j])] = -2
    var compact_vertices = 0
    for i in range(vertex_count):
        if mapping[i] == -2:
            mapping[i] = Int64(compact_vertices)
            if compact_vertices != i:
                store_v(points, compact_vertices, load_v(points, i))
                var k = 0
                while k + W <= attr_dim:
                    attrs.store(
                        compact_vertices * attr_dim + k,
                        attrs.load[width=W](i * attr_dim + k),
                    )
                    k += W
                while k < attr_dim:
                    attrs[compact_vertices * attr_dim + k] = attrs[i * attr_dim + k]
                    k += 1
            compact_vertices += 1
    var compact_faces = 0
    for face in range(face_count):
        if face_alive[face] != 0:
            for j in range(3):
                faces[3 * compact_faces + j] = Int32(
                    mapping[Int(faces[3 * face + j])]
                )
            compact_faces += 1
    result[0] = 0
    result[1] = Int64(compact_vertices)
    result[2] = Int64(compact_faces)
    result[3] = Int64(collapse_count)


def replay_core(
    points: FPtr,
    faces: JPtr,
    collapses: JPtr,
    q: FPtr,
    border: IPtr,
    parent: IPtr,
    active: IPtr,
    face_alive: IPtr,
    head: IPtr,
    next_ref: IPtr,
    mapping: IPtr,
    result: IPtr,
    vertex_count: Int,
    face_count: Int,
    collapse_count: Int,
):
    comptime W = simd_width_of[DType.float64]()

    for i in range(vertex_count):
        border[i] = 0
        parent[i] = Int64(i)
        active[i] = 1
        head[i] = -1
        mapping[i] = -1
        var k = 0
        while k + W <= 10:
            q.store(10 * i + k, SIMD[DType.float64, W](0.0))
            k += W
        while k < 10:
            q[10 * i + k] = 0.0
            k += 1

    for face in range(face_count):
        face_alive[face] = 1
        for j in range(3):
            var vertex = Int(faces[3 * face + j])
            var node = 3 * face + j
            next_ref[node] = head[vertex]
            head[vertex] = Int64(node)
        var a = load_v(points, Int(faces[3 * face]))
        var b = load_v(points, Int(faces[3 * face + 1]))
        var c = load_v(points, Int(faces[3 * face + 2]))
        var normal = (b - a).cross(c - a)
        var length = sqrt(normal.squared_norm())
        if length > 1.0e-30:
            normal = normal * (1.0 / length)
            var d = -normal.dot(a)
            for j in range(3):
                add_plane(
                    q,
                    Int(faces[3 * face + j]),
                    normal.x,
                    normal.y,
                    normal.z,
                    d,
                )

    for face in range(face_count):
        for j in range(3):
            var u = Int(faces[3 * face + j])
            var v = Int(faces[3 * face + (j + 1) % 3])
            if edge_face_count(faces, face_alive, head, next_ref, u, v) == 1:
                border[u] = 1
                border[v] = 1

    for collapse in range(collapse_count):
        var u = Int(collapses[2 * collapse])
        var v = Int(collapses[2 * collapse + 1])
        if (
            u < 0
            or v < 0
            or u >= vertex_count
            or v >= vertex_count
            or active[u] == 0
            or active[v] == 0
        ):
            result[0] = 1
            return
        var position, _, _ = candidate(points, q, border, u, v)
        store_v(points, u, position)
        var k = 0
        while k + W <= 10:
            q.store(
                10 * u + k,
                q.load[width=W](10 * u + k) + q.load[width=W](10 * v + k),
            )
            k += W
        while k < 10:
            q[10 * u + k] += q[10 * v + k]
            k += 1
        active[v] = 0
        parent[v] = Int64(u)

    for i in range(vertex_count):
        var root = i
        while Int(parent[root]) != root:
            root = Int(parent[root])
        mapping[i] = Int64(root)
        active[i] = 0

    var compact_faces = 0
    for face in range(face_count):
        var a = Int(mapping[Int(faces[3 * face])])
        var b = Int(mapping[Int(faces[3 * face + 1])])
        var c = Int(mapping[Int(faces[3 * face + 2])])
        if a != b and b != c and c != a:
            faces[3 * compact_faces] = Int32(a)
            faces[3 * compact_faces + 1] = Int32(b)
            faces[3 * compact_faces + 2] = Int32(c)
            active[a] = 1
            active[b] = 1
            active[c] = 1
            compact_faces += 1

    var compact_vertices = 0
    for i in range(vertex_count):
        if active[i] != 0:
            parent[i] = Int64(compact_vertices)
            if compact_vertices != i:
                store_v(points, compact_vertices, load_v(points, i))
            compact_vertices += 1
        else:
            parent[i] = -1

    for face in range(compact_faces):
        for j in range(3):
            faces[3 * face + j] = Int32(parent[Int(faces[3 * face + j])])

    for i in range(vertex_count):
        mapping[i] = parent[Int(mapping[i])]

    result[0] = 0
    result[1] = Int64(compact_vertices)
    result[2] = Int64(compact_faces)


@export("mfs_simplify")
def mfs_simplify(
    points_address: Int,
    faces_address: Int,
    attrs_address: Int,
    weights_address: Int,
    quadrics_address: Int,
    face_alive_address: Int,
    vertex_alive_address: Int,
    border_address: Int,
    head_address: Int,
    next_ref_address: Int,
    versions_address: Int,
    marks_address: Int,
    mapping_address: Int,
    heap_errors_address: Int,
    heap_us_address: Int,
    heap_vs_address: Int,
    heap_vus_address: Int,
    heap_vvs_address: Int,
    collapses_address: Int,
    result_address: Int,
    vertex_count: Int,
    face_count: Int,
    attr_dim: Int,
    target_faces: Int,
    max_error: Float64,
    heap_capacity: Int,
) abi("C"):
    simplify_core(
        fp(points_address),
        jp(faces_address),
        fp(attrs_address),
        fp(weights_address),
        fp(quadrics_address),
        ip(face_alive_address),
        ip(vertex_alive_address),
        ip(border_address),
        ip(head_address),
        ip(next_ref_address),
        ip(versions_address),
        ip(marks_address),
        ip(mapping_address),
        fp(heap_errors_address),
        up(heap_us_address),
        up(heap_vs_address),
        jp(collapses_address),
        ip(result_address),
        vertex_count,
        face_count,
        attr_dim,
        target_faces,
        max_error,
        heap_capacity,
    )


@export("mfs_replay")
def mfs_replay(
    points_address: Int,
    faces_address: Int,
    collapses_address: Int,
    quadrics_address: Int,
    border_address: Int,
    parent_address: Int,
    active_address: Int,
    face_alive_address: Int,
    head_address: Int,
    next_ref_address: Int,
    mapping_address: Int,
    result_address: Int,
    vertex_count: Int,
    face_count: Int,
    collapse_count: Int,
) abi("C"):
    replay_core(
        fp(points_address),
        jp(faces_address),
        jp(collapses_address),
        fp(quadrics_address),
        ip(border_address),
        ip(parent_address),
        ip(active_address),
        ip(face_alive_address),
        ip(head_address),
        ip(next_ref_address),
        ip(mapping_address),
        ip(result_address),
        vertex_count,
        face_count,
        collapse_count,
    )


@export("mfs_version")
def mfs_version() abi("C") -> Int:
    return 1
