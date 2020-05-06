import * as Dynamics from './dynamics.js';
import * as Geometry from './geometry.js';
import * as Simulation from './simulate.js';
import Vector from './vector.js';

const canvas = document.getElementById('physicsCanvas');
const ctx = canvas.getContext('2d');

const world = new Simulation.World(ctx, true);

world.addBody(new Dynamics.StaticBody(
    new Vector(250, 0), 
    new Geometry.Box(500, 1000, new Vector(0, -500))
));

world.addBody(new Dynamics.StaticBody(
    new Vector(250, 500), 
    new Geometry.Box(500, 1000, new Vector(0, 500))
));

world.addBody(new Dynamics.StaticBody(
    new Vector(0, 250), 
    new Geometry.Box(1000, 500, new Vector(-500, 0))
));

world.addBody(new Dynamics.StaticBody(
    new Vector(500, 250), 
    new Geometry.Box(1000, 500, new Vector(500, 0))
));

world.addBody(new Dynamics.DynamicBody(
    new Vector(250, 400),
    new Vector(0, 0),
    new Geometry.Box(30),
    new Dynamics.Material(1)
));
world.addBody(new Dynamics.DynamicBody(
    new Vector(250, 200),
    new Vector(0, 0),
    new Geometry.Circle(100),
    new Dynamics.Material(1)
));

world.addBody(new Dynamics.DynamicBody(
    new Vector(250, 0),
    new Vector(0, 0),
    new Geometry.Box(400, 100),
    new Dynamics.Material(.9)
));

world.run();