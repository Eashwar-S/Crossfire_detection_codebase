import math
def calcluatecoords():

    # Constants
    dFOV = CONSTANT * math.pi / 180 # degrees
    xPixMax = CONSTANT 
    yPixMax = CONSTANT
    rEarth = CONSTANT # meters

    # Variable
    height = VARIABLE # meters
    heading = VARIABLE * math.pi / 180 # north: 0, east: 90, south: 180, west: 270
    lat = VARIABLE # latitude
    lon = VARIABLE # longitude
    xPixVal = VARIABLE # pixel of interest location, 0: left, xPixMax: right
    yPixVal = VARIABLE # pixel of interest location, 0: up, yPixMax: down
    pitch = -VARIABLE * math.pi / 180 # pitch of the camera, facing down: -90, facing 45: -45, facing forward: 0

    # Calculating x and y FOV
    omega = math.atan(yPixMax/xPixMax)
    xFOV = math.cos(omega) * dFOV
    yFOV = math.sin(omega) * dFOV

    # Pixel -> Angle from the center
    xAngle = (xPixMax/2 - xPixVal) * xFOV/xPixMax
    yAngle = -(yPixMax/2 - yPixVal) * yFOV/yPixMax

    # Vector functions
    vn = (math.cos(pitch) - math.tan(yAngle) * math.sin(pitch)) * math.cos(heading) + math.tan(xAngle) * math.sin(heading)
    ve = (math.cos(pitch) - math.tan(yAngle) * math.sin(pitch)) * math.sin(heading) - math.tan(xAngle) * math.cos(heading)
    vd = math.sin(pitch) + math.tan(yAngle) * math.cos(pitch)

    # Displacement Calculations
    mn = height * vn/vd
    me = height * ve/vd


    # final equations
    latF = lat + (mn / 111132.0)
    lonF = lon + (me / (111132.0 * math.cos(lat * math.pi / 180)))

    # latF = lat + (diffY/(rEarth/2)) * 180
    # lonF = lon + (diffX/(rEarth/2)) * 180

    status_message = f"Estimated location: {latF}, {lonF}"
    return latF, lonF
