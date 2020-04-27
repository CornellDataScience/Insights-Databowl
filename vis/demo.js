import * as Dynamics from './physics/dynamics.js';
import * as Geometry from './physics/geometry.js';
import * as Simulation from './physics/simulate.js';
import Vector from './physics/vector.js';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const world = new Simulation.World(ctx, true);

function loadJSON(filePath, callback) {
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
            callback(JSON.parse(xhr.responseText))
        }
    };
    xhr.open("GET", filePath, true);
    xhr.send();
}

class PlayerBody extends Dynamics.DynamicBody {
    constructor(position, velocity, color) {
        const shape = new Geometry.Circle(3);
        const material = new Dynamics.Material(0.5);
        super(position, velocity, shape, material);

        this.hitboxColor = color;
        this.velocityColor = color;
        this.addCollisionHandler(() => {})
    }
}

loadJSON('play.json', play => {
    play.forEach(player => {
        const position = new Vector(player.X, (53.3-player.Y)).scale(8);
    
        const direction = (player.Dir + 90) % 360
        const velocity = new Vector(player.S * Math.cos(direction), -player.S * Math.sin(direction)).scale(8);
        const color = player.IsRusher ? 'green' : player.Team ? 'red' : 'blue';
    
        const body = new PlayerBody(position, velocity, color);
        world.addBody(body);
    
        console.log(body, player)
    });
});

// wall north
world.addBody(new Dynamics.StaticBody(
    new Vector(120/2, 0).scale(8), 
    new Geometry.Box(120*8, 100, new Vector(0, -50))
));

// wall south
world.addBody(new Dynamics.StaticBody(
    new Vector(120/2, 53.3).scale(8), 
    new Geometry.Box(120*8, 100, new Vector(0, 50))
));

// wall west
world.addBody(new Dynamics.StaticBody(
    new Vector(0, 53.3/2).scale(8),
    new Geometry.Box(100, 53.3*8, new Vector(-50, 0))
));

// wall east
world.addBody(new Dynamics.StaticBody(
    new Vector(120, 53.3/2).scale(8),
    new Geometry.Box(100, 53.3*8, new Vector(50, 0))
));

world.run();