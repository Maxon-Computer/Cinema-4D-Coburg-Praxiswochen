"""University Coburg Praxiswochen Prototyping Example.

Evaluates the "local thickness" of a polygon object by raycasting into the
shell of the object and weighting the retrieved intersection distances. The
resulting normalized data is written to a vertex map.

This example showcases how to use Cinema's Python bindings, specifically the
Python scripting tag, to prototype/flesh out a naive/first implementation of
an idea. Shown is also how to use the viewport drawing capabilities of the
Python API to help debugging such a prototype.

This project covers these subjects:
    * Handling of polygons, vertices and normals in Cinema
    * How to carry out some basic linear algebra in Cinema
        * Dot & cross product and normalization with c4d.Vector
        * Linear transforms/maps with c4d.Matrix
    * Tags and vertex maps
    * Raycasting via GeRayCollider
    * User data (and creating a GUI with them)
    * Drawing into viewports to convey information
    * Cinema's message system and DescIDs

Reading entry points:
    This file can be read in principle from top to bottom, but the entry
    points for Cinema 4D are the main(), message() and draw() functions at
    the bottom of the file.

Pseudo Code:
    This is the pseudo code for the core logic of the project.

    Let geom be a polygonal geometry.
    Let vertices be its list of vertices.
    Let theta be an angle within [0, +pi].
    Let count be the number of samples per vertex.
     
    weights = []
    for v in vertices:
        nrm = GetMeanVertexNormal(geom, v)
        samples = []
        for rayDirection in SolidAngle(v, -nrm, theta, count):
            distance = RayCast(geom, v, rayDirection)
            samples.append(distance)
        w = Integrate(samples)
        weights.append(w)
    vertexMap = CreateVertexMap(geom, distances)
"""

import c4d
import math
import statistics

# Some constants to make the code more readable, most of them are the IDs to
# access the user data.
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
TWO_PI = math.pi * 2. # Two pi


def Hash11(x, seed=1234., magic=(1234.5678, 98765.4321)):
    """Returns a pseudo random floating-point hash in the interval [0, 1]
    for a given floating-point number.

    Args:
        x (float): The value to get the pseudo random hash for.
        seed (float, optional): The seed offset for the hash.
        magic (tuple[float, float], optional): The magic numbers to massage
         the hash into place. Defaults to (1234.5678, 98765.4321).

    Returns:
        float: The random hash for x in the interval [0, 1].
    """
    return math.modf(math.sin((x * seed) * magic[0]) * magic[1])[0] * .5 + .5


def GetNormal(pid, points, polygons, neighbor):
    """Computes the mean normal for a given vertex.

    Note:
        You might want to read GetPointsAndNormals() below first.

    Args:
        pid (int): The point id to get the normal for.
        points (list[c4d.Vector]): The list of points of the node.
        polygons (list[c4d.CPolygon]): The list of polygons of the node.
        neighbor (c4d.utils.Neighbor): An adjacency data structure.

    Returns:
        c4d.Vector: The computed normal.
    """
    # Get the polygon ids attached to a vertex.
    connected = neighbor.GetPointPolys(pid)
    vertexNormals = []

    # Compute the vertex normal for each attached polygon.
    for cpoly in [polygons[pid] for pid in connected]:
        # Polygons are stored in Cinema with the CPolygon type, which
        # references four point indices in its fields a, b, c and d. Even
        # triangles are stored in this way. They just repeat their c index in
        # their d index.
        # We have then just to sort out which point index pairs form
        # the edges a and b in a cross-product operation to compute the
        # normal for a given vertex in a polygon.
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
        # We append the cross product of the edges a and b adjacent to
        # to the given vertex as its normal and normalize it. % is the
        # cross product operator and ~ the normalization operator.
        vNormal = ~((o-p) % (q-p))
        vertexNormals.append(vNormal)
    # Compute and return the mean normal.
    factor = 1./len(vertexNormals)
    return ~(sum(vertexNormals) * factor)


def GetPointsAndNormals(node):
    """Computes the mean vertex normals for a polygon object, ignoring
     phong breaks.

    Note:
        Cinema also has a method to retrieve the phong normals for a node,
        PolygonObject.CreatePhongNormals(), but in this case we would have
        to post-process that output anyways, so we are going to compute the
        normals ourselves.

    Args:
        node (c4d.PolygonObject): The object to get the normals for.

    Returns:
        tuple(list[c4d.Vector], list[c4d.Vector]): The points and vertex
         normal pairs for the node.
    """
    # Neighbor is an adjacency data structure allowing us to access adjacent
    # points, edges and polygons in a polygonal mesh. We have to initialize
    # it with the node we are interested in.
    neighbor = c4d.utils.Neighbor()
    neighbor.Init(node)
    # The points and polygons for the node.
    points = node.GetAllPoints()
    polygons = node.GetAllPolygons()
    data = []
    # Go over all vertices of the node and compute the mean normal for each.
    for pid, p in enumerate(points):
        n = GetNormal(pid, points, polygons, neighbor)
        data.append((p, n))
    # Free the adjacency data and return our points and normals.
    neighbor.Flush()
    return data


def NormalsInSolidAngle(normal, theta, samples):
    """Constructs unit vectors that lie within the solid angle specified by
     the given normal and angle.

    Yields:
        c4d.Vector: A unit vector that lies within the solid angle.

    Args:
        normal (c4d.Vector): The normal to align the solid angle with.
        theta (float): The angle of the solid angle.
        samples (tuple[int, int]): The longitudinal and latitudinal sample
         count.
    """
    def GetParametricCoordinates(theta, samples):
        """Build the parametric coordinates for sampling on a sphere section.
        """
        u, v = samples[0], samples[1]
        return [(float(iu) / float(u) * TWO_PI,
                 float(iv) / float(v) * theta)
                for iv in range(v)
                for iu in range(u)][u - 1:]

    # Construct a frame for the given normal. The * operator computes the dot
    # product between two vectors here (normal and c4d.Vector(0, 1, 0)).
    # There is also Vector.Dot() which can be used instead, e.g.,
    #   a.Dot(b) == a * b
    isParallel = abs(normal * c4d.Vector(0, 1, 0)) <= 1E-3
    up = c4d.Vector(0, 1, 0) if isParallel else c4d.Vector(0, 1.01, 0)
    # The ~ operator returns a normalized vector, e.g., ~v == v_hat and the
    # the % operator computes the cross product between two vectors. There
    # are also the methods Vector.GetNormalized() and Vector.Cross() which
    # can be used instead, but they tend to make the code hard to read IMHO.
    i = ~(normal % up)
    j = ~(i % normal)
    # The final transform. Cinema follows the convention of labeling the
    # basis vector i as v1, j as v2 and k as v3.
    normalFrame = c4d.Matrix(off=c4d.Vector(), v1=i, v2=j, v3=normal)

    # Compute and yield the final normals for the spherical coordinates. We
    # construct each normal in the world frame and then yield it transformed
    # into the frame we constructed for the input normal.
    for _theta, _phi in GetParametricCoordinates(theta, samples):
        # Spherical to Cartesian coordinates
        x = math.cos(_theta) * math.sin(_phi)
        y = math.sin(_theta) * math.sin(_phi)
        z = math.cos(_phi)
        # Yield the final normal and normalize it for good measure ;)
        yield ~(normalFrame * c4d.Vector(x, y, z))


def SampleDistance(p, normal, rayCaster, tag, debug=False):
    """Computes the approximated local thickness for a vertex and its mesh.

    Args:
        p (c4d.Vector): The point to for the vertex to sample.
        normal (c4d.Vector): The normal to for the vertex to sample.
        rayCaster (c4d.utils.GeRayCollider): A ray casting object.
        tag (c4d.BaseTag): The Python scripting tag holding the user data.
        debug (bool, optional): If True, the function won't normalize the
         distances and also will return the ray direction vectors. Used by
         the debug drawing functionality of draw().

    Returns:
        float or tuple[c4d.Vector, float]: The computed distances for p if
         debug is False. Otherwise the rays and normalized distances.
    """
    result = []
    sampleDistance = tag[ID_SAMPLES_DISTANCE]
    # For each ray direction within the solid angle aligned with the normal:
    for rayDirection in NormalsInSolidAngle(normal,
                                            theta=tag[ID_SAMPLES_ANGLE],
                                            samples=(tag[ID_SAMPLES_LONG],
                                                     tag[ID_SAMPLES_LAT])):
        # The ray origin for the ray casting operation. We nudge the ray
        # origin a little bit inwards to make it easier for us.
        rayOrigin = p + rayDirection * .05
        # We carry out the raycasting, GeRayCollider does all the heavy
        # lifting for us, we do not have to deal with the individual
        # triangles of our mesh.
        does_intersect = rayCaster.Intersect(rayOrigin, rayDirection,
                                             sampleDistance)
        dist = sampleDistance
        # We found no intersections, we simply assume that there is an
        # intersection beyond our maximum ray intersection distance and just
        # append that to our results.
        if not does_intersect:
            result.append((rayDirection, dist) if debug else dist)
            continue
        # We found intersections and now go through all of them to find the
        # shortest intersection distance that occurred.
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
        list[float]: The computed thickness weights in  [0, 1].
    """
    # Setup the ray casting object and some placeholder for the final data.
    rayCaster = c4d.utils.GeRayCollider()
    rayCaster.Init(node)
    distances = []
    # Some user data settings
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

    # Return the normalized weights. RangeMap() is a function which lets one
    # map a value x in the interval [a, b] to the value x' in the interval
    # [c, d]. What is nice about RangeMap() is that allows for passing a
    # SplineData instance (we have one as the Weighting attribute in our GUI)
    # to carry out the mapping in a non-linear fashion.
    return [c4d.utils.RangeMap(w, 0, max(distances), 0., 1., True, spline)
            for w in distances]


def CreateVertexmap():
    """Create a new vertex map.

    Creates a new vertex map tag for the polygon object hosting the Python
    tag containing this code, attaches it to the polygon object and links
    the tag in the user data of the Python tag.

    Returns:
        bool: If the operation has been successful.
    """
    # The attribute op is predefined in most scripting objects in Cinema 4D.
    # In case of a Python scripting tag it is a reference to the tag itself.

    # Get the object node the tag is attached to.
    node = op.GetObject()

    # Test if that node is an editable polygon object. We could also deal
    # with parametric polygon objects here, but for simplicity sake we will
    # restrict inputs to editable objects here.
    # We also test for being in the main thread since this is a requirement
    # for modifying the scene graph.
    if (not isinstance(node, c4d.PolygonObject) or
            not c4d.threading.GeIsMainThread()):
        return False

    # Create a vertex map with as many data points as the node has vertices.
    tag = c4d.VariableTag(c4d.Tvertexmap, node.GetPointCount())
    # We could have technically run out of memory here ;)
    if tag is None:
        return False

    # Attach the tag to the polygon object and link the tag in the user data.
    node.InsertTag(tag)
    op[ID_VERTEXMAP] = tag
    c4d.EventAdd()
    return True


def SetVertexMapData(tag, data):
    """Sets the data for a given vertex map.

    Args:
        tag (c4d.VariableTag): The vertex map to write the data to.
        data (list[float]): The data to write.

    Raises:
        RuntimeError: When the tags data count does not match the length of
         the passed data.
    """
    # Little sanity check for the passed data matching our vertex map data.
    if tag.GetDataCount() != len(data):
        raise RuntimeError("Unexpected computed data to vertex map data "
                           "count mismatch.")

    # Write the data into the tag.
    tag.SetAllHighlevelData(data)


def Execute():
    """Runs the core logic of the script.
    """
    # The attribute op is predefined in most scripting objects in Cinema 4D.
    # In case of a Python scripting tag it is a reference to the tag itself.

    # Get the object node the tag is attached to and test if the node is an
    # editable polygon object. We could also deal with parametric polygon
    # objects here, but for simplicity sake we will restrict inputs to
    # editable objects.
    node = op.GetObject()
    if not isinstance(node, c4d.PolygonObject):
        return False

    # Test for the Vertexmap user data element being populated with a vertex
    # map tag.
    vertexMap = op[ID_VERTEXMAP]
    if vertexMap is None or not vertexMap.CheckType(c4d.Tvertexmap):
        return False

    # Compute and write the thickness data.
    weights = GetThicknessWeigths(node, op)
    SetVertexMapData(vertexMap, data=weights)


def message(mid, data):
    """Retrieves messages sent to the node.

    Messages are used in Cinema 4D to convey event-like information. In this
    case we are listening for the buttons and sliders in the user data 
    interface being pressed/dragged.

    Args:
        mid (int): The message id.
        data (Any): The message data.

    Returns:
        Any: Depends on the message type.

    Raises:
        RuntimeError: When creating a new vertex map has failed.
    """
    def validateVertexMap():
        """Helper for verifying that the linked vertexmap is attached to the
        same object as the Python tag.
        """
        node = op.GetObject()
        tag = op[ID_VERTEXMAP]
        if (not isinstance(tag, c4d.BaseTag) or
            not tag.IsAlive() or
            not tag.CheckType(c4d.Tvertexmap) or
                not (tag.GetObject() == node)):
            op[ID_VERTEXMAP] = None

    # Make sure the linked vertex map is valid.
    validateVertexMap()
    # Test if this is a description command message, i.e. a message sent by
    # one of our interface buttons.
    if mid is c4d.MSG_DESCRIPTION_COMMAND:
        # Interface elements are identified in Cinema with something called
        # DescID which is basically just a collection of IDs to identify
        # attributes of a node. Here we are asking if the attribute that
        # raised this message has in its first ID level the user data ID and
        # in the second level the button ID we are looking for. The button
        # IDs for user data can be found in the user data interface and
        # depend on the order in which the user data elements have been
        # created.
        eid = (data["id"][0].id, data["id"][1].id)
        # The "Create" button has been pressed.
        if eid == ID_CREATE:
            if not CreateVertexmap():
                msg = ("Could not create vertex map. Tag is probably not "
                       "being hosted by an editable polygon object.")
                raise RuntimeError(msg)
        # The "Manual Update" button has been pressed.
        elif eid == ID_UPDATE:
            Execute()
    # Test if this is a description POSTSETPARAMETER message, i.e. a message 
    # sent by one of our interface elements after a value change.
    elif mid is c4d.MSG_DESCRIPTION_POSTSETPARAMETER:
        eid = (data["descid"][0].id, data["descid"][1].id)
        # The user changed the lower cutoff value. We want to prevent the
        # lower value being greater than the upper value.
        if eid == ID_INTEGRATION_LOWERCUTOFF:
            a = op[ID_INTEGRATION_LOWERCUTOFF]
            b = op[ID_INTEGRATION_UPPERCUTOFF]
            if a > b:
                op[ID_INTEGRATION_LOWERCUTOFF] = b
        # The user changed the upper cutoff value. We want to prevent the
        # upper value being smaller than the lower value.
        elif eid == ID_INTEGRATION_UPPERCUTOFF:
            a = op[ID_INTEGRATION_LOWERCUTOFF]
            b = op[ID_INTEGRATION_UPPERCUTOFF]
            if b < a:
                op[ID_INTEGRATION_UPPERCUTOFF] = a
    return True


def draw(bd):
    """Draws the debug info into a viewport.

    This is probably one of the diciest parts of this solution and we are
    here bending a bit the rules of what make sense to do in a Python tag.

    We are redoing part of our raycasting here for each drawn viewport frame.
    Which is of course bad. In a TagData plugin we could rely on cached data,
    but due to the volatile nature of scripting objects that is not possible
    here (at least not without doing hacky stuff like injecting cached data
    into places where we should not inject things).

    Args:
        bd (c4d.BaseDraw): The viewport to draw into.

    Returns:
        bool: Success of drawing into the viewport.
    """
    # Get out if there is not anything to draw.
    drawNormals = op[ID_DEBUG_NORMALS]
    drawRays = op[ID_DEBUG_RAYS]
    if not op[ID_DEBUG] or (not drawNormals and not drawRays):
        return True

    # Get out when we are not in the object draw pass. There are other draw
    # passes, but they do not lend themselves very well for what we are
    # drawing here.
    if bd.GetDrawPass() != c4d.DRAWPASS_OBJECT:
        return True

    # Get the node attached to the tag and test if its a polygon object.
    node = op.GetObject()
    if not isinstance(node, c4d.PolygonObject):
        return True

    # Get the point selection of the node and convert to vertex ids.
    # Selections are stored as a BaseSelect in Cinema 4D. They store a
    # list of boolean values for the selection states of what they represent
    # (vertices, edges or polygons). GetAll() below would return for
    # example [T, F, F, T] for a PointObject with its 0th and 3rd
    # vertex selected.
    selection = node.GetPointS()
    pointIDs = [i for i, v in enumerate(
        selection.GetAll(node.GetPointCount())) if v]
    # Bail if the selection is empty.
    if not pointIDs:
        return True

    # Set the drawing coordinate system to the coordinate system of the node,
    # so that we can draw with local point coordinates of the node.
    bd.SetMatrix_Matrix(node, node.GetMg())

    # We are not going to run our whole GetThicknessWeigths() here, since this
    # would be rather inefficient. Instead we are going to sample just what
    # we need. So we have to redo the Neighbor and GeRayCollider thing here.
    # Note: In case you are reading from bottom to top - these two types are
    # shown when you follow the call chain of GetThicknessWeigths().
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
        # Get the point and its normal.
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
            # Sample the distances with the optional debug argument so that
            # we aslo get the ray directions passed - which we do need here :)
            for rayDirection, distance in SampleDistance(
                    p, normal, rayCaster, op, debug=True):
                bd.DrawLine(p, p + rayDirection * distance,  c4d.NOCLIP_D)
    return True


def main():
    """Called by Cinema 4D when evaluating the Python tag.

    Cinema will call this function quite often, and not only when we change
    something in the GUI of the Python tag. This usually where most of your
    code goes for more straight forward Python tag uses cases. We only fire
    of the automatic updates from here.

    We could also throttle the overhead here a bit, but due to the volatile
    nature of a Python scripting tag this mostly falls into the domain of
    hacky. If a more sophisticated solution is required, one should move
    to full blown Python or C++ TagData plugin. Moving to a Python TagData
    plugin will be quite easy, due to the fact that we already did a good
    portion of the work.

    A C++ plugin will of course require a full rewrite, but having such quick
    and dirty Python tag prototype can also be quite helpful for writing a 
    C++ TagData plugin.
    """
    # Update the vertexmap when "Automatic Updates" is turned on.
    if op[ID_AUTOMATIC_UPDATES]:
        Execute()