/**
 * @file main.cpp
 * @brief This file contains the implementation of a 2D Curve Drawing Application using OpenGL.
 */

#include "framework.h"

/**
 * @brief The vertex shader in GLSL.
 */
 const char *const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

/**
 * @brief The fragment shader in GLSL.
 */
 const char *const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; /**< The GPU Program for vertex and fragment shaders. */

/**
 * @brief Class representing a 2D Camera.
 */
class Camera2D {
    vec2 wCenter; /**< The center in world coordinates. */
    vec2 wSize; /**< The width and height in world coordinates. */
public:

    /**
     * @brief Default constructor.
     */
    Camera2D() : wCenter(0.0f, 0.0f), wSize(30.0f, 30.0f) {}

    /**
     * @brief Computes the view matrix.
     * @return The view matrix.
     */
    mat4 V() { return TranslateMatrix(-wCenter); }

    /**
     * @brief Computes the projection matrix.
     * @return The projection matrix.
     */
    mat4 P() const { // projection matrix
        return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y));
    }

    /**
     * @brief Computes the inverse view matrix.
     * @return The inverse view matrix.
     */
    mat4 Vinv() { // inverse view matrix
        return TranslateMatrix(wCenter);
    }

    /**
     * @brief Computes the inverse projection matrix.
     * @return The inverse projection matrix.
     */
    mat4 Pinv() const { // inverse projection matrix
        return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2));
    }

    /**
     * @brief Zooms the camera by a given factor.
     * @param s The zoom factor.
     */
    void Zoom(float s) { wSize = wSize * s; }

    /**
     * @brief Pans the camera by a given translation vector.
     * @param t The translation vector.
     */
    void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D cam; /**< The 2D Camera object. */
const int nTverts = 100; /**< The number of vertices for drawing. */

/**
 * @brief Abstract base class for defining a curve.
 */
class Curve {
    unsigned int vao{}; /**< Vertex array object. */
    unsigned int vbo{}; /**< Vertex buffer object. */
protected:
    std::vector<vec2> points; /**< Vector storing control points of the curve. */
public:
    /**
    * @brief Constructor to initialize VAO and VBO.
    */
    Curve() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    }

    /**
     * @brief Destructor to delete VAO and VBO.
     */
    ~Curve() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }

    /**
     * @brief Pure virtual function to compute the position of the curve at a given parameter t.
     * @param t The parameter.
     * @return The position of the curve at parameter t.
     */
    virtual vec2 r(float t) = 0;

    /**
     * @brief Pure virtual function to get the start parameter of the curve.
     * @return The start parameter of the curve.
     */
    virtual float tStart() = 0;

    /**
     * @brief Pure virtual function to get the end parameter of the curve.
     * @return The end parameter of the curve.
     */
    virtual float tEnd() = 0;

    /**
     * @brief Adds a control point to the curve.
     * @param cX The X coordinate of the control point.
     * @param cY The Y coordinate of the control point.
     */
    virtual void addP(float cX, float cY) {
        points.emplace_back((vec4(cX, cY, 0, 1) * cam.Pinv() * cam.Vinv()).x,
                            (vec4(cX, cY, 0, 1) * cam.Pinv() * cam.Vinv()).y);
    }

    /**
     * @brief Picks a control point from the curve.
     * @param cX The X coordinate of the cursor.
     * @param cY The Y coordinate of the cursor.
     * @return The index of the picked control point.
     */
    int pickP(float cX, float cY) {
        vec2 vw = vec2((vec4(cX, cY, 0, 1) * cam.Pinv() * cam.Vinv()).x,
                       (vec4(cX, cY, 0, 1) * cam.Pinv() * cam.Vinv()).y);
        int p = 0;
        while (p < points.size()) {
            if (dot(points[p] - vw, points[p] - vw) < 0.1) {
                return p;
            }
            p++;
        }
        return -1;
    }

    /**
     * @brief Moves a control point of the curve.
     * @param p The index of the control point.
     * @param cX The X coordinate of the cursor.
     * @param cY The Y coordinate of the cursor.
     */
    void moveP(int p, float cX, float cY) {
        points[p] = vec2((vec4(cX, cY, 0, 1) * cam.Pinv() * cam.Vinv()).x,
                         (vec4(cX, cY, 0, 1) * cam.Pinv() * cam.Vinv()).y);
    }

    /**
     * @brief Draws the curve.
     */
    void Draw() {
        gpuProgram.setUniform(cam.V() * cam.P(), "MVP");

        if (!points.empty()) {
            std::vector<vec2> vData(nTverts);
            if (points.size() >= 2) {
                auto it = vData.begin();
                while (it != vData.end()) {
                    float tNorm = (float) std::distance(vData.begin(), it) / (nTverts - 1);
                    float t = tStart() + (tEnd() - tStart()) * tNorm;
                    vec2 wVertex = r(t);
                    *it = wVertex;
                    ++it;
                }
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vData.size()) * sizeof(vec2), &vData[0],
                             GL_DYNAMIC_DRAW);
                gpuProgram.setUniform(vec3(1, 1, 0), "color");
                glLineWidth(2.0f);
                glDrawArrays(GL_LINE_STRIP, 0, nTverts);
            }

            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(points.size()) * sizeof(vec2), &points[0],
                         GL_STATIC_DRAW);
            glPointSize(10.0f);
            gpuProgram.setUniform(vec3(1, 0, 0), "color");
            glDrawArrays(GL_POINTS, 0, (GLsizeiptr) points.size());
        }
    }
};

/**
 * @brief Class representing a Bezier curve, derived from Curve.
 */
class BezierCurve : public Curve {

    /**
    * @brief Computes the binomial coefficient (n choose i).
    * @param n The total number of elements.
    * @param i The number of elements to choose.
    * @return The binomial coefficient.
    */
    static float choose(int n, int i) {
        float choose = 1.0;
        int j = 1;
        while (j <= i) {
            choose *= (float) (n - j + 1) / (float) j;
            j++;
        }
        return choose;
    }

    /**
     * @brief Computes the Bernstein polynomial.
     * @param it Iterator to the control point.
     * @param t The parameter value.
     * @return The value of the Bernstein polynomial at parameter t.
     */
    float B(std::vector<vec2>::iterator it, float t) {
        int n = std::distance(points.begin(), points.end()) - 1;
        int i = std::distance(points.begin(), it);
        return static_cast<float>(choose(n, i) * pow(1 - t, n - i) * pow(t, i));
    }

public:
    /**
     * @brief Computes the start parameter value of the Bezier curve.
     * @return The start parameter value.
     */
    float tStart() override { return 0; }

    /**
     * @brief Computes the end parameter value of the Bezier curve.
     * @return The end parameter value.
     */
    float tEnd() override { return 1; }

    /**
     * @brief Computes the position of the Bezier curve at a given parameter t.
     * @param t The parameter value.
     * @return The position of the Bezier curve at parameter t.
     */
    vec2 r(float t) override {
        vec2 wPoint(0, 0);
        auto it = points.begin();
        while (it != points.end()) {
            wPoint = wPoint + (*it) * B(it, t);
            ++it;
        }
        return wPoint;
    }
};

/**
 * @brief Class representing a Lagrange curve, derived from Curve.
 */
class LagrangeCurve : public Curve {
    std::vector<float> p; /**< Parameter values for the Lagrange curve. */

    /**
     * @brief Calculates the Lagrange interpolation coefficient Li.
     * @param t The parameter value.
     * @param it_i Iterator to the current control point.
     * @param it_j Iterator to another control point.
     * @return The Lagrange interpolation coefficient.
     */
    float calculateLi(float t, std::vector<vec2>::iterator it_i, std::vector<vec2>::iterator it_j) {
        if (it_j != it_i) {
            return (t - p[std::distance(points.begin(), it_j)]) /
                   (p[std::distance(points.begin(), it_i)] - p[std::distance(points.begin(), it_j)]);
        }
        return 1.0f;
    }

    /**
     * @brief Calculates the Lagrange interpolation coefficient L.
     * @param it_i Iterator to the current control point.
     * @param t The parameter value.
     * @return The Lagrange interpolation coefficient.
     */
    float L(std::vector<vec2>::iterator it_i, float t) {
        float Li = 1.0f;
        auto it_j = points.begin();
        while (it_j != points.end()) {
            Li *= calculateLi(t, it_i, it_j);
            ++it_j;
        }
        return Li;
    }

public:
    /**
     * @brief Computes the start parameter value of the Lagrange curve.
     * @return The start parameter value.
     */
    void addP(float cX, float cY) override {
        Curve::addP(cX, cY);
        calts();
    }

    /**
     * @brief Computes the start parameter value of the Lagrange curve.
     * @return The start parameter value.
     */
    static float calculate(const vec2 &point1, const vec2 &point2) {
        return (float) std::sqrt(std::pow(point2.x - point1.x, 2) + std::pow(point2.y - point1.y, 2));
    }

    /**
     * @brief Computes the total distance between the control points.
     * @param distances The vector storing the distances between the control points.
     * @return The total distance between the control points.
     */
    static float calculateTotalDistance(const std::vector<float> &distances) {
        float total = 0.0f;
        auto it = distances.begin();
        while (it != distances.end()) {
            total += *it;
            ++it;
        }
        return total;
    }

    /**
     * @brief Computes the parameter values for the Lagrange curve.
     */
    void calts() {
        p.clear();
        p.push_back(0.0f);
        std::vector<float> distances(points.size() - 1);

        auto it1 = points.begin();
        auto it2 = distances.begin();
        while (it1 != points.end() - 1) {
            *it2 = calculate(*it1, *(it1 + 1));
            ++it1;
            ++it2;
        }

        float total = calculateTotalDistance(distances);

        float ds = 0.0f;
        auto it3 = distances.begin();
        while (it3 != distances.end()) {
            ds += *it3;
            p.push_back(ds / total);
            ++it3;
        }
    }

    /**
     * @brief Computes the start parameter value of the Lagrange curve.
     * @return The start parameter value.
     */
    float tStart() override { return p[0]; }

    /**
     * @brief Computes the end parameter value of the Lagrange curve.
     * @return The end parameter value.
     */
    float tEnd() override { return p[points.size() - 1]; }

    /**
     * @brief Computes the position of the Lagrange curve at a given parameter t.
     * @param t The parameter value.
     * @return The position of the Lagrange curve at parameter t.
     */
    vec2 calculatePoint(std::vector<vec2>::iterator it_i, float t) {
        return *it_i * L(it_i, t);
    }

    /**
     * @brief Computes the position of the Lagrange curve at a given parameter t.
     * @param t The parameter value.
     * @return The position of the Lagrange curve at parameter t.
     */
    vec2 r(float t) override {
        vec2 pw(0, 0);
        auto it_i = points.begin();
        while (it_i != points.end()) {
            pw = pw + calculatePoint(it_i, t);
            ++it_i;
        }
        return pw;
    }
};

/**
 * @brief Class representing a Catmull-Rom spline, derived from Curve.
 */
class CatmullRomSpline : public Curve {
    float tension = 0.0f; /**< Tension parameter for controlling the shape of the spline. */
    std::vector<float> p; /**< Parameter values for the spline segments. */

    /**
     * @brief Computes the Hermite interpolation between two points with given tangents and parameter values.
     * @param p0 The first point.
     * @param v0 The tangent at the first point.
     * @param t0 The parameter value corresponding to the first point.
     * @param p1 The second point.
     * @param v1 The tangent at the second point.
     * @param t1 The parameter value corresponding to the second point.
     * @param t The parameter value at which to evaluate the Hermite interpolation.
     * @return The interpolated point.
     */
    static vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
        return ((2.0f * (p0 - p1) / (float) pow(t1 - t0, 3)) + ((v1 + v0) / (float) pow(t1 - t0, 2))) *
               (float) pow(t - t0, 3)
               + ((3.0f * (p1 - p0) / (float) pow(t1 - t0, 2)) - ((v1 + 2 * v0) / (t1 - t0))) * (float) pow(t - t0, 2) +
               (v0 * (t - t0))
               + p0;
    }

public:
    /**
     * @brief Overrides the addP function to recalculate parameter values after adding a control point.
     * @param cX The x-coordinate of the control point.
     * @param cY The y-coordinate of the control point.
     */
    void addP(float cX, float cY) override {
        Curve::addP(cX, cY);
        calts();
    }

    /**
     * @brief Computes the distance between two points.
     * @param point1 The first point.
     * @param point2 The second point.
     * @return The distance between the two points.
     */
    static float calculate(const vec2 &point1, const vec2 &point2) {
        return (float) std::sqrt(std::pow(point2.x - point1.x, 2) + std::pow(point2.y - point1.y, 2));
    }

    /**
     * @brief Computes the total distance between the control points.
     * @param distances The vector storing the distances between the control points.
     * @return The total distance between the control points.
     */
    static float calculateTotalDistance(const std::vector<float> &distances) {
        float total = 0.0f;
        auto it = distances.begin();
        while (it != distances.end()) {
            total += *it;
            ++it;
        }
        return total;
    }

    /**
     * @brief Computes the parameter values for the Catmull-Rom spline.
     */
    void calts() {
        p.clear();
        p.push_back(0.0f);
        std::vector<float> distances(points.size() - 1);

        auto it1 = points.begin();
        auto it2 = distances.begin();
        while (it1 != points.end() - 1) {
            *it2 = calculate(*it1, *(it1 + 1));
            ++it1;
            ++it2;
        }

        float total = calculateTotalDistance(distances);

        float ds = 0.0f;
        auto it3 = distances.begin();
        while (it3 != distances.end()) {
            ds += *it3;
            p.push_back(ds / total);
            ++it3;
        }
    }

    /**
     * @brief Computes the start parameter value of the spline.
     * @return The start parameter value.
     */
    float tStart() override { return p[0]; }

    /**
    * @brief Computes the end parameter value of the spline.
    * @return The end parameter value.
    */
    float tEnd() override { return p[points.size() - 1]; }


    /**
     * @brief Calculates the tangent at the start point of a spline segment.
     * @param it_i Iterator to the current control point.
     * @return The tangent vector.
     */
    vec2 calcZ(std::vector<vec2>::iterator it_i) {
        vec2 temp;
        if (it_i == points.begin()) {
            temp = ((*(it_i + 1) - *it_i) /
                    (p[std::distance(points.begin(), it_i) + 1] - p[std::distance(points.begin(), it_i)]) +
                    vec2(0.0, 0.0));
        } else {
            temp = ((*(it_i + 1) - *it_i) /
                    (p[std::distance(points.begin(), it_i) + 1] - p[std::distance(points.begin(), it_i)]) +
                    (*it_i - *(it_i - 1)) /
                    (p[std::distance(points.begin(), it_i)] - p[std::distance(points.begin(), it_i) - 1]));
        }
        return temp * ((1.0f - tension) / 2.0f);
    }
    /**
    * @brief Calculates the tangent at the end point of a spline segment.
    * @param it_i Iterator to the current control point.
    * @return The tangent vector.
    */
    vec2 calcF(std::vector<vec2>::iterator it_i) {
        vec2 temp;
        if (it_i == points.end() - 2) {
            temp = (vec2(0.0, 0.0) + (*(it_i + 1) - *it_i) / (p[std::distance(points.begin(), it_i) + 1] -
                                                              p[std::distance(points.begin(), it_i)]));
        } else {
            temp = ((*(it_i + 2) - *(it_i + 1)) /
                    (p[std::distance(points.begin(), it_i) + 2] - p[std::distance(points.begin(), it_i) + 1]) +
                    (*(it_i + 1) - *it_i) /
                    (p[std::distance(points.begin(), it_i) + 1] - p[std::distance(points.begin(), it_i)]));
        }
        return temp * ((1.0f - tension) / 2.0f);
    }

    /**
    * @brief Calculates a point on the spline using Hermite interpolation.
    * @param it_i Iterator to the current control point.
    * @param t The parameter value.
    * @return The interpolated point.
    */
    vec2 calculateHermite(std::vector<vec2>::iterator it_i, float t) {
        vec2 vZ = calcZ(it_i);
        vec2 vF = calcF(it_i);
        return Hermite(*it_i, vZ, p[std::distance(points.begin(), it_i)], *(it_i + 1), vF,
                       p[std::distance(points.begin(), it_i) + 1], t);
    }

    /**
     * @brief Computes a point on the spline at parameter t.
     * @param t The parameter value.
     * @return The point on the spline at parameter t.
     */
    vec2 r(float t) override {
        auto it_i = points.begin();
        while (it_i != points.end() - 1) {
            if (p[std::distance(points.begin(), it_i)] <= t && t <= p[std::distance(points.begin(), it_i) + 1]) {
                return calculateHermite(it_i, t);
            }
            ++it_i;
        }
        return points[0];
    }

    /**
     * @brief Increases the tension parameter.
     */
    void IncreaseTension() {
        tension += 0.1f;
    }

    /**
     * @brief Decreases the tension parameter.
     */
    void DecreaseTension() {
        tension -= 0.1f;
    }
};

Curve *curve; /**< The currently selected curve object. */

/**
 * @brief Sets up the viewport.
 */
void setupViewport() {
    glViewport(0, 0, windowWidth, windowHeight);
}

/**
 * @brief Sets up point size and line width.
 */
void setupPointSizeAndLineWidth() {
    glPointSize(10.0f);
    glLineWidth(2.0f);
}

/**
 * @brief Initializes the curve object.
 */
void initializeCurve() {
    curve = new LagrangeCurve();
}

/**
 * @brief Creates the GPU program.
 */
void createGPUProgram() {
    gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

/**
 * @brief Initialization function called once.
 */
void onInitialization() {
    setupViewport();
    setupPointSizeAndLineWidth();
    initializeCurve();
    createGPUProgram();
}

/**
 * @brief Sets the clear color for the frame buffer.
 */
void setClearColor() {
    glClearColor(0, 0, 0, 0);     // background color
}

/**
 * @brief Clears the frame buffer.
 */
void clearFrameBuffer() {
    GLenum mask = GL_COLOR_BUFFER_BIT;
    mask |= GL_DEPTH_BUFFER_BIT;
    glClear(mask);
}

/**
 * @brief Draws the curve.
 */
void drawCurve() {
    curve->Draw();
}

/**
 * @brief Swaps the buffers.
 */
void swapBuffers() {
    glutSwapBuffers();
}

/**
 * @brief Display function called when the screen needs to be redrawn.
 */
void onDisplay() {
    setClearColor();
    clearFrameBuffer();
    drawCurve();
    swapBuffers();
}

/**
 * @brief Keyboard function called on key press.
 * @param key The ASCII code of the pressed key.
 * @param pX The X coordinate of the cursor.
 * @param pY The Y coordinate of the cursor.
 */
 void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'l') {
        curve = new LagrangeCurve();
    } else if (key == 'b') {
        curve = new BezierCurve();
    } else if (key == 'c') {
        curve = new CatmullRomSpline();
    } else if (key == 'z') {
        cam.Zoom(1.0f / 1.1f);
    } else if (key == 'Z') {
        cam.Zoom(1.1f);
    } else if (key == 'P') {
        cam.Pan(vec2(1, 0));
    } else if (key == 'p') {
        cam.Pan(vec2(-1, 0));
    } else if (key == 'T') {
        ((CatmullRomSpline *) curve)->IncreaseTension();
    } else if (key == 't') {
        ((CatmullRomSpline *) curve)->DecreaseTension();
    } else {
        return;
    }
    glutPostRedisplay();
}

/**
 * @brief Keyboard function called on key release.
 * @param key The ASCII code of the released key.
 * @param pX The X coordinate of the cursor.
 * @param pY The Y coordinate of the cursor.
 */
 void onKeyboardUp(unsigned char key, int pX, int pY) {
}

int pickedControlPoint = -1;

/**
 * @brief Mouse motion function called when the mouse moves with a key pressed.
 * @param pX The X coordinate of the cursor.
 * @param pY The Y coordinate of the cursor.
 */
 void onMouseMotion(int pX,
                   int pY) {    // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    float cX = 2.0f * (float) pX / windowWidth - 1;    // flip y axis
    float cY = 1.0f - 2.0f * (float) pY / windowHeight;
    if (pickedControlPoint >= 0) {
        curve->moveP(pickedControlPoint, cX, cY);
    }
    glutPostRedisplay();
}


/**
 * @brief Mouse function called on mouse click.
 * @param button The mouse button pressed.
 * @param state The state of the button (pressed or released).
 * @param pX The X coordinate of the cursor.
 * @param pY The Y coordinate of the cursor.
 */
 void onMouse(int button, int state, int pX, int pY) {
    float cX = 2.0f * (float) pX / windowWidth - 1;    // flip y axis
    float cY = 1.0f - 2.0f * (float) pY /
                      windowHeight;// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        curve->addP(cX, cY);
        glutPostRedisplay();
    }
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        pickedControlPoint = curve->pickP(cX, cY);
    }
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
        pickedControlPoint = -1;
    }
}

/**
 * @brief Idle function called when the application is idle.
 */
 void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}



