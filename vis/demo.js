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
    constructor(player) {
        // get data
        const position = new Vector(player.X, (53.3-player.Y)).scale(8);
    
        const direction = (player.Dir + 90) % 360
        const velocity = new Vector(player.S * Math.cos(direction), -player.S * Math.sin(direction)).scale(8);

        const weight = player.PlayerWeight;

        // set attributes
        const shape = new Geometry.Circle(3);
        const material = new Dynamics.Material(0.2, 1, weight);
        super(position, velocity, shape, material);

        this.player = player;

        this.color = player.IsRusher ? 'green' : player.Offense ? 'red' : 'blue';;
        this.hitboxColor = this.color;
        this.velocityColor = this.color;
      //  this.addCollisionHandler(() => {})
    }

    get acceleration() {
        if (this.player.Offense) {
            return new Vector(this.player.A, 0).scale(8)
        }
        return super.acceleration;
    }

    draw(ctx) {
        ctx.beginPath();
        ctx.arc(this.shape.x, this.shape.y, this.shape.r, 0, 2*Math.PI);
        ctx.fillStyle = this.color;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(this.shape.x, this.shape.y, this.shape.r*2, 0, 2*Math.PI);
        ctx.strokeStyle = this.color;
        ctx.stroke();
    }
}

loadJSON('play.json', play => {
    const metadata = play[0];

    world.draw = (ctx) => {
        // endzones
        ctx.fillStyle = 'black';
        ctx.fillRect(10*8, 0, 1, 120*8);
        ctx.fillRect(110*8, 0, 1, 120*8);

        // scrimage
        ctx.fillStyle = 'red';
        ctx.fillRect((110-metadata.YardLine)*8, 0, 1, 120*8);

        // yards
        ctx.fillStyle = 'blue';
        ctx.fillRect((110-metadata.YardLine+metadata.Yards)*8, 0, 1, 120*8);
    };

    play.forEach(player => {
        const body = new PlayerBody(player);
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