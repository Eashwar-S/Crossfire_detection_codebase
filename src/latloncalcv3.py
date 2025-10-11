import math
def calcluatecoords(xPixVal, yPixVal):

    # Constants
    dFOV = 54.143 * math.pi / 180 # degrees, 54.143 for 1.7x zoom, 82 for 1x zoom
    xPixMax = 720 
    yPixMax = 640
    height = CONSTANT # Put height of the drone during the flight in meters
    pitch = CONSTANT * math.pi / 180 - 0.000001 # INPUT pitch of the camera, facing down: 90, facing 45: 45, facing forward: 0

    # Variable
    heading = _latest[heading] * math.pi / 180 - 0.000001 # north: 0, east: 90, south: 180, west: 270
    lat = _latest["air_lat"] # latitude
    lon = _latest["air_lon"] # longitude

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
