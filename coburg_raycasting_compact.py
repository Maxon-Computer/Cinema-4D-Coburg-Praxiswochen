"""University Coburg Praxiswochen Prototyping Example.

Evaluates the "local thickness" of a polygon object by raycasting into the
shell of the object and weighting the retrieved intersection distances. The
resulting normalized data is written to a vertex map.
"""

import c4d
import math
import statistics

# Some constants to make the code more readable.
ID_VERTEXMAP = (c4d.ID_USERDATA, 1) # The vertexmap
ID_CREATE = (c4d.ID_USERDATA, 2) # The create button
ID_INTEGRATION_METHOD = (c4d.ID_USERDATA, 3) # The integration method id
 # The lower cutoff for the samples. If the value for example is .2 and we
 # have the samples [1, 2, 3, 4, 5], only [2, 3, 4, 5] will be integrated.
ID_INTEGRATION_LOWERCUTOFF = (c4d.ID_USERDATA, 4)
ID_INTEGRATION_UPPERCUTOFF = (c4d.ID_USERDATA, 5) # The upper cutoff
ID_SAMPLES_DISTANCE = (c4d.ID_USERDATA, 6) # The sample distance
ID_SAMPLES_LONG = (c4d.ID_USERDATA, 7) # The longitudinal sample count
ID_SAMPLES_LAT = (c4d.ID_USERDATA, 8) # The latitudinal sample count
# The angle of the solid angle to cast in.
ID_SAMPLES_ANGLE = (c4d.ID_USERDATA, 9)
ID_WEIGHTING = (c4d.ID_USERDATA, 10) # The weighting SplineData
ID_AUTOMATIC_UPDATES = (c4d.ID_USERDATA, 11) # The automatic updates state
ID_UPDATE = (c4d.ID_USERDATA, 12) # The update button
ID_DEBUG = (c4d.ID_USERDATA, 13) # The debug state
ID_DEBUG_NORMALS = (c4d.ID_USERDATA, 14) # Debug normals
ID_DEBUG_RAYS = (c4d.ID_USERDATA, 15) # Debug rays
# The cheapskate integration functions ;)
INTEGRATION_FUNCTIONS = {
    0: statistics.mean,
    1: statistics.harmonic_mean,
    2: statistics.median,
    3: min,
    4: max,
    5: sum
}
TWO_PI = math.pi * 2.


def Hash11(x, seed=1234., magic=(1234.5678, 98765.4321)):
    """Returns a pseudo random floating-point hash in the interval [0, 1].

    Args:
        x (float): The value to get the pseudo random hash for.
        seed (float, optional): The seed offset for the hash.
        magic (tuple[float, float], optional): The magic numbers.
         Defaults to (1234.5678, 98765.4321).

    Returns:
        float: The random hash for x in the interval [0, 1].
    """
    return math.modf(math.sin((x * seed) * magic[0]) * magic[1])[0] * .5 + .5


def GetNormal(pid, points, polygons, neighbor):
    """Computes the mean normal for a given vertex.

    Args:
        pid (int): The point id to get the normal for.
        points (list[c4d.Vector]): The list of points of the node.
        polygons (list[c4d.CPolygon]): The list of polygons of the node.
        neighbor (c4d.utils.Neighbor): An adjacency data structure.

    Returns:
        c4d.Vector: The computed normal.
    """
    connected = neighbor.GetPointPolys(pid)
    vertexNormals = []

    for cpoly in [polygons[pid] for pid in connected]:
        if pid == cpoly.a:
            o, p, q = points[cpoly.d], points[cpoly.a], points[cpoly.b]
        elif pid == cpoly.b:
            o, p, q = points[cpoly.a], points[cpoly.b], points[cpoly.c]
        elif pid == cpoly.c and cpoly.IsTriangle():
            o, p, q = points[cpoly.b], points[cpoly.c], points[cpoly.a]
        elif pid == cpoly.c and not cpoly.IsTriangle():
            o, p, q = points[cpoly.b], points[cpoly.c], points[cpoly.d]
        elif pid == cpoly.d:
            o, p, q = points[cpoly.c], points[cpoly.d], points[cpoly.a]
        vNormal = ~((o-p) % (q-p))
        vertexNormals.append(vNormal)
    factor = 1./len(vertexNormals)
    return ~(sum(vertexNormals) * factor)


def GetPointsAndNormals(node):
    """Computes the mean vertex normals for a polygon object, ignoring
     phong breaks.

    Args:
        node (c4d.PolygonObject): The object to get the normals for.

    Returns:
        tuple(list[c4d.Vector], list[c4d.Vector]): The points and  normals.
    """
    neighbor = c4d.utils.Neighbor()
    neighbor.Init(node)
    points = node.GetAllPoints()
    polygons = node.GetAllPolygons()
    data = []
    for pid, p in enumerate(points):
        n = GetNormal(pid, points, polygons, neighbor)
        data.append((p, n))
    neighbor.Flush()
    return data


def NormalsInSolidAngle(normal, theta, samples):
    """Constructs unit vectors that lie within a specified solid angle.

    Yields:
        c4d.Vector: A unit vector that lies within the solid angle.

    Args:
        normal (c4d.Vector): The normal to align the solid angle with.
        theta (float): The angle of the solid angle.
        samples (tuple[int, int]): The sample count.
    """
    def GetParametricCoordinates(theta, samples):
        """Build the parametric coordinates for sampling on a sphere section.
        """
        u, v = samples[0], samples[1]
        return [(float(iu) / float(u) * TWO_PI,
                 float(iv) / float(v) * theta)
                for iv in range(v)
                for iu in range(u)][u - 1:]

    # Construct a frame for the given normal.
    isParallel = abs(normal * c4d.Vector(0, 1, 0)) <= 1E-3
    up = c4d.Vector(0, 1, 0) if isParallel else c4d.Vector(0, 1.01, 0)
    i = ~(normal % up)
    j = ~(i % normal)
    normalFrame = c4d.Matrix(off=c4d.Vector(), v1=i, v2=j, v3=normal)

    # Compute and yield the final normals for the spherical coordinates.
    for _theta, _phi in GetParametricCoordinates(theta, samples):
        x = math.cos(_theta) * math.sin(_phi)
        y = math.sin(_theta) * math.sin(_phi)
        z = math.cos(_phi)
        yield ~(normalFrame * c4d.Vector(x, y, z))


def SampleDistance(p, normal, rayCaster, tag, debug=False):
    """Computes the approximated local thickness for a vertex and its mesh.

    Args:
        p (c4d.Vector): The point to for the vertex to sample.
        normal (c4d.Vector): The normal to for the vertex to sample.
        rayCaster (c4d.utils.GeRayCollider): A ray casting object.
        tag (c4d.BaseTag): The Python scripting tag holding the user data.
        debug (bool, optional): Private.

    Returns:
        float or tuple[c4d.Vector, float]: The computed distances for p.
    """
    result = []
    sampleDistance = tag[ID_SAMPLES_DISTANCE]
    # For each ray direction within the solid angle:
    for rayDirection in NormalsInSolidAngle(normal,
                                            theta=tag[ID_SAMPLES_ANGLE],
                                            samples=(tag[ID_SAMPLES_LONG],
                                                     tag[ID_SAMPLES_LAT])):
        rayOrigin = p + rayDirection * .05
        does_intersect = rayCaster.Intersect(rayOrigin, rayDirection,
                                             sampleDistance)
        dist = sampleDistance

        if not does_intersect:
            result.append((rayDirection, dist) if debug else dist)
            continue
        for index in range(rayCaster.GetIntersectionCount()):
            rayData = rayCaster.GetIntersection(index)
            if (rayData["backface"] == True and
                    rayData["distance"] < dist):
                dist = rayData["distance"]
        result.append((rayDirection, dist) if debug else dist)
    return result


def GetThicknessWeigths(node, tag):
    """Computes the local thickness weight data for a given node.

    This is the entry point for the core logic of the project.

    Args:
        node (c4d.PolygonObject): The object to compute the data for.
        tag (c4d.BaseTag): The Python scripting tag holding the user data.

    Returns:
        list[float]: The computed thickness weights in [0, 1].
    """
    rayCaster = c4d.utils.GeRayCollider()
    rayCaster.Init(node)
    distances = []

    integrationMethod = tag[ID_INTEGRATION_METHOD]
    cuttoffA = tag[ID_INTEGRATION_LOWERCUTOFF]
    cuttoffB = tag[ID_INTEGRATION_UPPERCUTOFF]
    spline = tag[ID_WEIGHTING]

    # Go over all point/normal pairs for the object.
    for p, n in GetPointsAndNormals(node):
        # Sample the local thickness for that point/vertex.
        samples = SampleDistance(p, n, rayCaster, tag)

        # Apply the post processing.
        if not (cuttoffA == .0 and cuttoffB == 1.):
            samples.sort()
            count = len(samples)
            a = int(count * cuttoffA)
            b = int(count * cuttoffB)
            if a == b and a == 0:
                b = 1
            elif a == b:
                a = b - 1
            samples = samples[a:b]

        # Compute the final singular distance value for the samples.
        distances.append(INTEGRATION_FUNCTIONS[integrationMethod](samples))

    # Return the normalized weights.
    return [c4d.utils.RangeMap(w, 0, max(distances), 0., 1., True, spline)
            for w in distances]


def CreateVertexmap():
    """Create a new vertex map.

    Returns:
        bool: If the operation has been successful.
    """
    node = op.GetObject()
    if (not isinstance(node, c4d.PolygonObject) or
            not c4d.threading.GeIsMainThread()):
        return False

    tag = c4d.VariableTag(c4d.Tvertexmap, node.GetPointCount())
    if tag is None:
        return False

    node.InsertTag(tag)
    op[ID_VERTEXMAP] = tag
    c4d.EventAdd()
    return True


def SetVertexMapData(tag, data):
    """Sets the data for a given vertex map.

    Args:
        tag (c4d.VariableTag): The vertex map to write the data to.
        data (list[float]): The data to write.
    """
    if tag.GetDataCount() != len(data):
        raise RuntimeError("Unexpected computed data to vertex map data "
                           "count mismatch.")

    tag.SetAllHighlevelData(data)


def Execute():
    """Runs the core logic of the script.
    """
    node = op.GetObject()
    if not isinstance(node, c4d.PolygonObject):
        return False

    # Test for the Vertexmap user data element being populated.
    vertexMap = op[ID_VERTEXMAP]
    if vertexMap is None or not vertexMap.CheckType(c4d.Tvertexmap):
        return False

    # Compute and write the thickness data.
    weights = GetThicknessWeigths(node, op)
    SetVertexMapData(vertexMap, data=weights)


def message(mid, data):
    """Retrieves messages sent to the node.
    """
    def validateVertexMap():
        """Helper for verifying that the linked vertexmap.
        """
        node = op.GetObject()
        tag = op[ID_VERTEXMAP]
        if (not isinstance(tag, c4d.BaseTag) or
            not tag.IsAlive() or
            not tag.CheckType(c4d.Tvertexmap) or
                not (tag.GetObject() == node)):
            op[ID_VERTEXMAP] = None

    validateVertexMap()

    # Buttons
    if mid is c4d.MSG_DESCRIPTION_COMMAND:
        eid = (data["id"][0].id, data["id"][1].id)

        if eid == ID_CREATE:
            if not CreateVertexmap():
                msg = ("Could not create vertex map. Tag is probably not "
                       "being hosted by an editable polygon object.")
                raise RuntimeError(msg)
        elif eid == ID_UPDATE:
            Execute()
    # Sliders
    elif mid is c4d.MSG_DESCRIPTION_POSTSETPARAMETER:
        eid = (data["descid"][0].id, data["descid"][1].id)

        # Clamp the cutoffs so that lower cannot exceed upper
        if eid == ID_INTEGRATION_LOWERCUTOFF:
            a = op[ID_INTEGRATION_LOWERCUTOFF]
            b = op[ID_INTEGRATION_UPPERCUTOFF]
            if a > b:
                op[ID_INTEGRATION_LOWERCUTOFF] = b
        elif eid == ID_INTEGRATION_UPPERCUTOFF:
            a = op[ID_INTEGRATION_LOWERCUTOFF]
            b = op[ID_INTEGRATION_UPPERCUTOFF]
            if b < a:
                op[ID_INTEGRATION_UPPERCUTOFF] = a
    return True


def draw(bd):
    """Draws the debug info into a viewport.
    """
    drawNormals = op[ID_DEBUG_NORMALS]
    drawRays = op[ID_DEBUG_RAYS]
    if not op[ID_DEBUG] or (not drawNormals and not drawRays):
        return True

    if bd.GetDrawPass() != c4d.DRAWPASS_OBJECT:
        return True

    node = op.GetObject()
    if not isinstance(node, c4d.PolygonObject):
        return True

    selection = node.GetPointS()
    pointIDs = [i for i, v in enumerate(
        selection.GetAll(node.GetPointCount())) if v]

    if not pointIDs:
        return True

    bd.SetMatrix_Matrix(node, node.GetMg())

    # We are not going to run our whole GetThicknessWeigths() here, since this
    # would be rather inefficient. Instead we are going to sample just what
    # we need.
    neighbor = c4d.utils.Neighbor()
    neighbor.Init(node)
    points = node.GetAllPoints()
    polygons = node.GetAllPolygons()
    rayCaster = c4d.utils.GeRayCollider()
    rayCaster.Init(node)
    # The color for normals
    normalColor = c4d.Vector(1)

    # For each selected vertex index.
    for pid in pointIDs:
        p = points[pid]
        normal = GetNormal(pid, points, polygons, neighbor)

        # Draw normals.
        if drawNormals:
            bd.SetPen(normalColor)
            bd.DrawLine(p, p - normal * 10, c4d.NOCLIP_D)

        # Draw rays.
        if drawRays:
            rayColor = c4d.Vector(Hash11(pid),
                                  Hash11(pid + 1),
                                  Hash11(pid + 2))
            bd.SetPen(rayColor)
            for rayDirection, distance in SampleDistance(
                    p, normal, rayCaster, op, debug=True):
                bd.DrawLine(p, p + rayDirection * distance,  c4d.NOCLIP_D)
    return True


def main():
    """Called by Cinema 4D when evaluating the Python tag.
    """
    if op[ID_AUTOMATIC_UPDATES]:
        Execute()