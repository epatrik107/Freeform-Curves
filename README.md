# Freeform Curves Editor

This project is a Freeform Curves Editor implemented in world coordinates. It supports Lagrange, Bézier, and Catmull-Rom splines. The entire parameter range for each curve is the [0,1] interval. For Lagrange and Catmull-Rom curves, this is divided into knot values such that the difference between consecutive knot values is proportional to the distance between control points.

## Features

- Point size: 10
- Line thickness: 2
- Control points color: Maximum intensity red
- Curve color: Maximum intensity yellow
- The viewport completely covers the 600x600 resolution application window.
- The distance unit in the virtual world is [m] (meter).

Initially, the center of the camera window is at the origin of the world coordinate system, and its size is 30x30 [m]. The user can change the camera window by pressing keys, panning it, and zooming in/out:

- 'Z': Increases the size of the camera window by 1.1 times while keeping the center (zoom-out).
- 'z': Decreases the size of the camera window to 1/1.1 times while keeping the center (zoom-in).
- 'P': Moves the camera window 1 meter to the right (pan).
- 'p': Moves the camera window 1 meter to the left (pan).

Pressing these keys immediately changes the image of the current curve according to the new camera window.

The type of the next defined curve can be determined by pressing the following keys:

- 'l': Lagrange
- 'b': Bézier
- 'c': Catmull-Rom

Pressing these keys destroys the current curve, if it exists, and you can start specifying the control points of the new curve. The control points of the curve are placed under the cursor when the left mouse button is pressed, i.e., the input pipeline must produce the inverse of the output transformation. A nearby (closer than 10 centimeters), already existing control point can be selected by pressing the right mouse button. The selected control point follows the cursor until the right button is released. 

With the 'T' key, the tension parameter of the current and future Catmull-Rom curves can be increased by 0.1, and with the 't' key, it can be decreased by the same amount. The shape of the curve immediately follows the change of the control point and tension parameter, i.e., it must be redrawn at this time.
