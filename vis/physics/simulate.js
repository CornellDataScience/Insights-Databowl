import * as Dynamics from './dynamics.js';
import * as Geometry from './geometry.js';
import Vector from './vector.js';

class Layer {
    constructor() {
        this.id = Layer.unique;
    }

    static get unique() {
        return 0; //todo
    }
}

export class World {
    /**
     * @param {*} ctx the canvas drawing context.
     * @param {boolean} debug whether to run in debug mode.
     * @param {number} fps maximum frames per second.
     * @param {number} cps maximum calculations per second (i.e. simulation granularity).
     */
    constructor(ctx, debug=false, fps=60, cps=200) {
        // rendering properties
        this.ctx = ctx;
        this.debug = debug;

        // body managers
        this.staticBodies = [];
        this.kinematicBodies = [];
        this.dynamicBodies = [];

        // time stepping properies
        this.fps = fps;
        this.dt = 1 / cps;
        this.acc = 0;
        this.stepLimit = 10; // maximum nuumber of steps per update
    }

    get allBodies() {
        return this.staticBodies.concat(this.kinematicBodies).concat(this.dynamicBodies);
    }

    /**
     * Add a new body to the world.
     * @param {Dynamics.Body} body
     */
    addBody(body) {
        if (body instanceof Dynamics.DynamicBody) {
            this.dynamicBodies.push(body);
        }
        else if (body instanceof Dynamics.StaticBody) {
            this.staticBodies.push(body);
        }
        else {
            this.kinematicBodies.push(body);
        }
    }

    /**
     * Collects all pairs of bodies that are close enough to collide.
     * Note that this list may have false positives but never false negatives.
     * @returns {Body[][]} a list of pairs of bodies that could potentially collide.
     */
    broadPhaseCollision() {
        const pairs = [];
        for (let i = 0; i < this.allBodies.length; i++) {
            for (let j = i+1; j < this.allBodies.length; j++) {
                const a = this.allBodies[i], b = this.allBodies[j];
                if (a instanceof Dynamics.DynamicBody || b instanceof Dynamics.DynamicBody) {
                    pairs.push([a, b]);
                }
            }
        }
        return pairs;
    }

    /**
     * Brute force resolves the collisions between all pairs of bodies in a list.
     * @param {Body[][]} bodyPairs
     */
    narrowPhaseCollision(bodyPairs) {
        bodyPairs.forEach(pair => {
            const a = pair[0], b = pair[1];
            const collision = new Dynamics.Collision(a, b);
            if (collision.detect()) {
                collision.resolve();
            }
        });
    }

    /**
     * Detects and resolves every dynamic collision.
     */
    resolveCollisions() {
        this.narrowPhaseCollision(this.broadPhaseCollision());
    }

    /**
     * Resolves collisions and updates the simulated bodies by `dt`.
     * @param {Number} dt change in time.
     */
    step(dt) {
        // play logic
        const defense = this.dynamicBodies.filter(b => b.color == 'blue')
        const offense = this.dynamicBodies.filter(b => b.color != 'blue')
        defense.forEach(d => {
            // find closest offensive player
            let min_d = 0;
            let min_p = null;
            for (let i = 0; i < offense.length; i++) {
                let o = offense[i];
                let distance = d.position.distance(o.position)
                if (min_p == null || distance < min_d) {
                    min_d = distance;
                    min_p = offense[i]
                }
            }

            // move defense towards offense
            const f = min_p.position.subtract(d.position).unit.scale(8)
            d.addForce(f.scale(d.mass * 4));
        })

        // integrate forces to produce new velocities
        this.dynamicBodies.forEach(b => {
            b.velocity = b.velocity.add(b.acceleration.scale(dt));
            b.force = Vector.ZERO;
        })

        // resolve collisions
        this.resolveCollisions();

        // update positions of bodies
        this.kinematicBodies.forEach(e => e.update(dt));
        this.dynamicBodies.forEach(e => e.update(dt));
    }

    /**
     * Performs `this.cps` simulation steps per second to update the simulation.
     * @param {*} t time of the simulation in seconds.
     */
    update(t) {
        // update the accumulator
        if (this.lastUpdate != undefined) {
            this.acc += t - this.lastUpdate;
        }
        this.lastUpdate = t;

        // clamp upper value of accumulator to reduce number of steps when too much load
        const accMax = this.dt * this.stepLimit;
        if (this.acc > accMax) {
            console.log('Simulation throttled:', this.acc - accMax, 'second delay.')
            this.acc = accMax;
        }

        // step the simulation in discrete `dt` sized chunks of time
        for (; this.acc >= this.dt; this.acc -= this.dt) {
            this.step(this.dt);
        }
    }

    /*
     *  World rendering
     */
    draw(ctx) { }

    /**
     * Renders every body onto the cavnas using their `.draw()` method.
     */
    render(t) {
        // linear interpolation using remaining accumulator value TODO
        const interpolation = this.acc / this.dt;

        this.ctx.clearRect(0, 0, 1000, 1000);
        this.draw(this.ctx)

        this.allBodies.forEach(e => {
            e.draw(this.ctx);
            if (this.debug) {
                e.trace(this.ctx);
            }
        });
    }

    /**
     * Launch simulation event loop.
     * @param {*} t 
     */
    run(t) {
        if (t == undefined) {
            t = performance.now();
        }
        t /= 1000; // scale time to seconds

        this.render(t);
        this.update(t);
        requestAnimationFrame(t => this.run(t));
    }
}