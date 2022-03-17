function init_overview() {
    $.ajax({
        type: "GET",
        url: "/overview/getPatientOverview",
    }).done(function (response) {

        if (typeof response === "object"){
            var title = document.getElementById("project_title");
            while (title.firstChild) {
                title.removeChild(title.firstChild);
            }
            var text = '<p><h2 class="panel-heading">WSI & ANNOTATION OVERVIEW</h2></p><div class="row img-rounded">\n' +
                '    <hr><small>\n' +
                '<div class="col-md-4">'
            var el = $(text);
            $("#project_title").append(el);
            document.getElementById("project_title").setAttribute("align", "center");
            $('div#project_title').css({'width': '100%'});
            createVisualization(response);
        }
    });
};

var vis;
var partition;
var arc;


var colors = {
    "Patient_HUPM": "#51a17e",
    "Patient_HUPM_shine": "#51a17e",
    "Patient_HUPR": "#F7EC5B",
    "Patient_HUPR_shine": "#F7EC5B",
    "Patient_HL": "#F2A86D",
    "Patient_HL_shine": "#F2A86D",
    "Patient_HJF": "#F26D6D",
    "Patient_HJF_shine": "#F26D6D",
    "HE": "#d171a7",
    "IMM": "#3a87ad",
    "HE_0": "#d0cdb0",
    "HE_00": "#d0cdb0",
    "HE_1": "#d171a7",
    "HE_11": "#d171a7",
    "HE_10": "#d171a7",
    "IMM_0": "#d0cdb0",
    "IMM_00": "#d0cdb0",
    "IMM_1": "#3a87ad",
    "IMM_11": "#3a87ad",
    "IMM_10": "#3a87ad",
};

var colors_stroke = {
    "Patient_HUPM": "none",
    "Patient_HUPM_shine": "black",
    "Patient_HUPR": "none",
    "Patient_HUPR_shine": "black",
    "Patient_HL": "none",
    "Patient_HL_shine": "black",
    "Patient_HJF": "none",
    "Patient_HJF_shine": "black",
    "HE_00": "none",
    "HE_11": "black",
    "HE_10": "none",
    "IMM_00": "none",
    "IMM_11": "black",
    "IMM_10": "none",
};

var legend_labels = {
    "Patient_HUPM": "Patient",
    "Patient_HUPM_shine": "Patient",
    "Patient_HUPR": "Patient",
    "Patient_HUPR_shine": "Patient",
    "Patient_HL": "Patient",
    "Patient_HL_shine": "Patient",
    "Patient_HJF": "Patient",
    "Patient_HJF_shine": "Patient",
    "HE": "HE",
    "IMM": "IMM",
    "HE_0": "Not annotated HE",
    "HE_00": "Not annotated HE",
    "HE_1": "Annotated HE",
    "HE_11": "Annotated & validated HE",
    "HE_10": "Annotated HE",
    "IMM_0": "Not annotated IMM",
    "IMM_00": "Not annotated IMM",
    "IMM_1": "Annotated IMM",
    "IMM_10": "Annotated IMM",
    "IMM_11": "Annotated & validated IMM",
}

// Dimensions of sunburst.
var width = $(window).height()*0.6;
var height = $(window).height()*0.6;
var radius = Math.min(width, height) / 2;
// Breadcrumb dimensions: width, height, spacing, width of tip/tail.
var b = {
    w: 180, h: 30, s: 3, t: 10
};


// Main function to draw and set up the visualization, once we have the data.
function createVisualization(json) {


    vis = d3.select("#chart").append("svg:svg")
        .attr("width", $('div#project_title').width())
        .attr("height", height)
        .append("svg:g")
        .attr("id", "container")
        .attr("transform", "translate(" + $('div#project_title').width() / 2 + "," + height / 2 + ")");



    partition = d3.layout.partition()
        .size([2 * Math.PI, radius * radius])
        .value(function (d) {
            return d.size;
        });

    arc = d3.svg.arc()
        .startAngle(function (d) {
            return d.x;
        })
        .endAngle(function (d) {
            return d.x + d.dx;
        })
        .innerRadius(function (d) {
            return Math.sqrt(d.y);
        })
        .outerRadius(function (d) {
            if (d.depth==3){
                return Math.sqrt(d.y + d.dy/2);
            }else{
                return Math.sqrt(d.y + d.dy);
            }
            //return Math.sqrt(d.y + d.dy);
        });
    // Basic setup of page elements.
    initializeBreadcrumbTrail();

    // Bounding circle underneath the sunburst, to make it easier to detect
    // when the mouse leaves the parent g.
    vis.append("svg:circle")
        .attr("r", radius)
        .style("opacity", 0);

    // For efficiency, filter nodes to keep only those large enough to see.
    nodes = partition.nodes(json)
        .filter(function (d) {
            return (d.dx > 0.0001); // 0.005 radians = 0.29 degrees
        });

    path = vis.data([json]).selectAll("path")
        .data(nodes)
        .enter().append("svg:path")
        .attr("display", function (d) {
            return d.depth ? null : "none";
        })
        .attr("d", arc)
        .attr("fill-rule", "evenodd")
        .style("fill", function (d) {
            if (d.depth==2)
            {
                return tinycolor.mix(colors[d.name], colors["HE_0"], amount = d.code*100)
            }else{
                return colors[d.code];
            }
        })
        .style("stroke", function (d) {

            if (d.depth == 3) {
                return colors_stroke[d.code]
            }
            if (d.depth == 1){
                return colors_stroke[d.code]
            }
            else{
                return "none";
            }
        })
        .style("opacity", 1)
        .on("mouseover", mouseover);

    // Add the mouseleave handler to the bounding circle.
    d3.select("#container").on("mouseleave", mouseleave);
};

// Fade all but the current sequence, and show it in the breadcrumb trail.
function mouseover(d) {

    var sequenceArray = getAncestors(d);
    // Fade all the segments.
    d3.selectAll("path")
        .style("opacity", 0.3);

    // Then highlight only those that are an ancestor of the current segment.
    vis.selectAll("path")
        .filter(function (node) {
            return (sequenceArray.indexOf(node) >= 0);
        })
        .style("opacity", 1);
    updateBreadcrumbs(sequenceArray);
}

// Restore everything to full opacity when moving off the visualization.
function mouseleave(d) {

    // Hide the breadcrumb trail
    d3.select("#trail")
        .style("visibility", "hidden");

    // Deactivate all segments during transition.
    d3.selectAll("path").on("mouseover", null);

    // Transition each segment to full opacity and then reactivate it.
    d3.selectAll("path")
        .transition()
        .duration(50)
        .style("opacity", 1)
        .each("end", function () {
            d3.select(this).on("mouseover", mouseover);
        });
}

// Given a node in a partition layout, return an array of all of its ancestor
// nodes, highest first, but excluding the root.
function getAncestors(node) {
    var path = [];
    var current = node;
    while (current.parent) {
        path.unshift(current);
        current = current.parent;
    }
    return path;
}

function initializeBreadcrumbTrail() {
    // Add the svg area.
    var trail = d3.select("#sequence").append("svg:svg")
        .attr("width", width)
        .attr("height", (50 + b.s) * 3)
        .attr("id", "trail");

}

// Generate a string that describes the points of a breadcrumb polygon.
function breadcrumbPoints(d, i) {
    var points = [];
    points.push("0,0");
    points.push(b.w + ",0");
    points.push(b.w + b.t + "," + (b.h / 2));
    points.push(b.w + "," + b.h);
    points.push("0," + b.h);
    if (i > 0) { // Leftmost breadcrumb; don't include 6th vertex.
        points.push(b.t + "," + (b.h / 2));
    }
    var out = points.join(" ");
    return out;
}

// Update the breadcrumb trail to show the current sequence and percentage.
function updateBreadcrumbs(nodeArray) {

    // Data join; key function combines name and depth (= position in sequence).
    var g = d3.select("#trail")
        .selectAll("g")
        .data(nodeArray, function (d) {
            return d.name + d.depth + d.code;
        });


    // Add breadcrumb and label for entering nodes.
    var entering = g.enter().append("svg:g");

    entering.append("svg:polygon")
        .attr("points", breadcrumbPoints)
        .style("fill", function (d) {
            if (d.depth==2)
            {
                return tinycolor.mix(colors[d.name], colors["HE_0"], amount = d.code*100)
            }else{
                return colors[d.code];
            }
        });

    entering.append("svg:text")
        .attr("x", (b.w + b.t) + b.s * 2)
        .attr("y", b.h / 2)
        .attr("dy", "0.35em")
        .attr("text-anchor", "left")
        .text(function (d) {
            if (d.depth==2)
            {
                return  ((1-d.code)*100).toFixed(2) + "%" + " annotated"
            }else{
                return d.name;
            }

        });
    entering.append("svg:text")
        .attr("x", (b.w + b.t) / 2)
        .attr("y", b.h / 2)
        .attr("dy", "0.35em")
        .attr("text-anchor", "middle")
        .text(function (d) {
            if (d.depth==2)
            {
                return d.name
            }else{
                return legend_labels[d.code];
            }

        });

    // Set position for entering and updating nodes.
    g.attr("transform", function (d, i) {
        return "translate(0, " + i * (b.h + b.s) + ")";
    });

    // Remove exiting nodes.
    g.exit().remove();

    // Make the breadcrumb trail visible, if it's hidden.
    d3.select("#trail")
        .style("visibility", "");

}
