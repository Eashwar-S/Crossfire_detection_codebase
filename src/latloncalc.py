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
    yPixVal = VARIABLE # pixel of interest location, 0: down, yPixMax: top
    pitch = VARIABLE * math.pi / 180 # pitch of the camera, facing down: 90, facing 45: 45, facing forward: 0

    # Calculating x and y FOV
    omega = math.atan(yPixMax/xPixMax)
    xFOV = math.cos(omega) * dFOV
    yFOV = math.sin(omega) * dFOV

    # Pixel -> Angle from the center
    xAngle = -(xPixMax/2 - xPixVal) * xFOV/xPixMax
    yAngle = -(yPixMax/2 - yPixVal) * yFOV/yPixMax

    # Vector functions
    d = (math.cos(yAngle - pitch))**2 + (math.cos(xAngle))**2  - ((math.cos(xAngle))**2) * ((math.cos(yAngle - pitch))**2)
    vX = (math.cos(yAngle - pitch)) * (math.sin(xAngle)) / (math.sqrt(d))
    vY = (math.cos(yAngle - pitch)) * (math.cos(xAngle)) / (math.sqrt(d))
    vZ = (math.sin(yAngle - pitch)) * (math.cos(xAngle)) / (math.sqrt(d))

    # Displacement Calculations
    dist_Right = -(vX * height / vZ)
    dist_Forward = (vY * height / -vZ)

    meters_North = dist_Forward * math.cos(heading) - dist_Right * math.sin(heading)
    meters_East  = dist_Forward * math.sin(heading) + dist_Right * math.cos(heading)

    # final equations
    latF = lat + (meters_North / 111132.0)
    lonF = lon + (meters_East / (111320.0 * math.cos(lat * math.pi / 180)))

    # latF = lat + (diffY/(rEarth/2)) * 180
    # lonF = lon + (diffX/(rEarth/2)) * 180

    status_message = f"Estimated location: {latF}, {lonF}"
    return latF, lonF
