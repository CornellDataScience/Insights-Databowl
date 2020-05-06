export default class Vector {
    constructor(x, y) {
        this.coords = [x, y];
    }

    set x(x) {
        this.coords[0] = x;
    }

    get x() {
        return this.coords[0];
    }

    set y(y) {
        this.coords[1] = y;
    }

    get y() {
        return this.coords[1];
    }

    get unit() {
        return this.magnitude == 1 ? this : this.apply(e => e / this.magnitude);
    }

    get magnitude() {
        return Math.hypot(...this.coords);
    }

    /**
     * Returns the angle of the vector in degrees in the interval [0, 360).
     * 0 degrees is along the positive x-axis and rotates counter-clockwise.
     */
    get angle() {
        const theta = Math.atan(this.y / this.x) * 180 / Math.PI;
        // quadrant 1
        if (Math.sign(this.x) >= 0 && Math.sign(this.y) >= 0) {
            return theta;
        }
        // quadrants 2 and 3
        else if (Math.sign(this.x) < 0) {
            return theta + 180;
        }
        // quadrant 4
        else {
            return theta + 360;
        }
    }

    add(other) {
        return Vector.combine(this, other, (a, b) => a + b);
    }

    subtract(other) {
        return Vector.combine(this, other, (a, b) => a - b);
    }

    scale(c) {
        return this.apply(e => e * c);
    }

    dot(other) {
        return Vector.combine(this, other, (a, b) => a * b).aggregate((sum, e) => sum + e);
    }

    distance(other) {
        return this.subtract(other).magnitude;
    }

    apply(f) { 
        return new Vector(...this.coords.map(f));
    }

    aggregate(f) { 
        return this.coords.reduce(f);
    }
    
    static get ZERO() {
        return new Vector(0, 0);
    }

    static combine(v1, v2, f) {
        return v1.apply((e, i) => f(e, v2.coords[i]));
    }
}