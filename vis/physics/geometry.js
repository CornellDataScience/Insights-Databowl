import Vector from './vector.js';

export class Shape {
    constructor(offset=Vector.ZERO) { 
        this.offset = offset;        // relative coordinates
        this.center = this.offset;    // global coordinates
    }

    get area() {
        return 0;
    }

    get x() {
        return this.center.x;
    }

    get y() {
        return this.center.y;
    }

    /**
     * @param {*} vector global location of this shape.
     */
    relocate(vector) {
        this.center = this.offset.add(vector);
    }

    /**
     * Creates a canvas path depicting this shape.
     * @param {*} ctx the canvas drawing context.
     */
    createPath(ctx) {
        return
    }
}

export class Point extends Shape {
    constructor() {
        super(Vector.ZERO)
    }
}

export class Circle extends Shape {
    constructor(radius, offset) {
        super(offset);
        this.r = radius;
    }

    get area() {
        return Math.PI * (this.r ** 2);
    }

    createPath(ctx) {
        ctx.arc(this.x, this.y, this.r, 0, 2*Math.PI);
    }
}

export class Box extends Shape {
    constructor(width, height, offset) {
        super(offset)
        this.w = width, this.h = height;
        if (height == undefined) {
            this.h = width;
        }
    }

    get area() {
        return this.w * this.h;
    }

    get min() {
        return new Vector(this.x - this.w/2, this.y - this.h/2)
    }

    get max() {
        return new Vector(this.x + this.w/2, this.y + this.h/2)
    }

    createPath(ctx) {
        ctx.rect(this.min.x, this.min.y, this.w, this.h);
    }
}

const clamp = (x, min, max) => {
    if (x < min){
        x = min;
    }
    else if (x > max) {
        x = max;
    }
    return x;
}

/**
 * Computes the contact manifold between two shapes.
 * @param {Shape} a end point of the vector.
 * @param {Shape} b start point of the vector. 
 * 
 * @returns {Vector} A collision vector from `b` to `a`.
 *                      The normal points in the direction of the transfer of energy.
 *                      The magnitude describes the penetration of the intersection.
 *                      Returns `Vector.ZERO` if shapes do not intersect.
 */
export function getManifold(a, b) {
    if (a instanceof Point && b instanceof Circle || a instanceof Circle && b instanceof Point) {
        const [p, c] = a instanceof Point ? [a, b] : [b, a];
        d = p.center.subtract(c.center)   // vector from circle to point
        if (d.magnitude <= c.r) {
            return d.scale(c.r - d.magnitude);
        }
    }

    else if (a instanceof Point && b instanceof Box || a instanceof Box && b instanceof Point) {
        const [p, r] = a instanceof Point ? [a, b] : [b, a];
        const overlap = new Vector()
    }

    else if (a instanceof Circle && b instanceof Circle) {
        const r = a.r + b.r;                    // sum of radii
        const d = a.center.distance(b.center)   // distance between centers

        if (r >= d) {
            if (d != 0) {
                //this.normal = a.center.subtract(b.center);
                return a.center.subtract(b.center).unit.scale(r - d);
            }
            else { // center of circles at same location
                //this.normal = new Vector(Math.random(), Math.random());
                return new Vector(Math.random(), Math.random()).unit.scale(r - d);
            }
        }
    }

    else if (a instanceof Box && b instanceof Box) {
        const d = a.center.subtract(b.center)   // vector between centers
        const overlap = new Vector(a.w/2 + b.w/2 - Math.abs(d.x), a.h/2 + b.h/2 - Math.abs(d.y))
        
        if (overlap.x >=0 && overlap.y >= 0) {
            if (overlap.x < overlap.y) {
                if (d.x > 0) {
                    return new Vector(overlap.x, 0)
                }
                else {
                    return new Vector(-overlap.x, 0)
                }
            }
            else {
                if (d.y > 0) {
                    return new Vector(0, overlap.y)
                }
                else {
                    return new Vector(0, -overlap.y)
                }
            }
        }
    }
    
    else if (a instanceof Box && b instanceof Circle || a instanceof Circle && b instanceof Box) {
        const [r, c] = a instanceof Box ? [a, b] : [b, a];
        const d = c.center.subtract(r.center)   // vector between centers
        // vector from center of rect to closest point on rect to center of circle
        let closest = new Vector(clamp(d.x, -r.w/2, r.w/2), clamp(d.y, -r.h/2, r.h/2))
        
        const inside = (d.subtract(closest).magnitude == 0);    // circle center inside rect

        if (inside) {
            if (Math.abs(d.x) > Math.abs(d.y)) {
                closest.x = r.w/2 * Math.sign(closest.x)
            }
            else {
                closest.y = r.h/2 * Math.sign(closest.y)
            }
        }

        // vector from center of circle to closest point on rect
        const normal = d.subtract(closest);
        if (normal.magnitude <= c.r || inside) {
            const vector = d.subtract(closest).scale((inside ? 1 : -1) * (a instanceof Box ? 1 : -1))
            return vector.scale(c.r - normal.magnitude)
        }
    }

    return Vector.ZERO;
}