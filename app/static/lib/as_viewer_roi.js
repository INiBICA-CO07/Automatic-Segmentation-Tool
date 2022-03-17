//(function() {                 // force everything local.
var debug = 1;

var ImageInfo = {};             // regions, and projectID (for the paper.js canvas) for each slices, can be accessed by
                                // the slice name. (e.g. ImageInfo[imageOrder[viewer.current_page()]])
                                // regions contain a paper.js path, a unique ID and a name
var currentImage = undefined;   // name of the current image
var currentStaining = undefined;
var slideName = "";   // name of the current image
var prevImage = undefined;      // name of the last image
var region = null;	            // currently selected region (one element of Regions[])
var handle;			            // currently selected control point or handle (if any)
var selectedTool;	            // currently selected tool
var viewer;			            // open seadragon viewer
var navEnabled = true;          // flag indicating whether the navigator is enabled (if it's not, the annotation tools are)
var magicV = 1000;	            // resolution of the annotation canvas - is changed automatically to reflect the size of the tileSource
var myOrigin = {};	            // Origin identification for DB storage
var params;			            // URL parameters
var newRegionFlag;	            // true when a region is being drawn

var annotationLoadingFlag;      // true when an annotation is being loaded
var config = {};                 // App configuration object
var isMac = navigator.platform.match(/Mac/i) ? true : false;
var isIOS = navigator.platform.match(/(iPhone|iPod|iPad)/i) ? true : false;

var currentAnnotation = "";


var buttonMode = false;
var prevGuiButton = "";

/***
 AnnotationService variables
 */
var tmpTool;

var staticPath;                 // path to flasks static folder
var slide;                      // slide object with (file-)name, url and mpp
var labelDictionary = [];       // dictionary with labels
var dictionaries = [];          // list of dictionaries
var currentLabel = {label: "no label"};  // currently selected label
var countAll = 0;
var countLabel = {};
var labelIds = {};


/*** Funciones WSI-staining
 */
function getNameFromCurrentStaining() {
    var label_dialog;

    if (currentStaining == 1)
        label_dialog = "HE";
    else if (currentStaining == 2)
        label_dialog = "ER";
    else if (currentStaining == 3)
        label_dialog = "PR";
    else if (currentStaining == 4)
        label_dialog = "AR";
    else if (currentStaining == 5)
        label_dialog = "p53";
    else if (currentStaining == 6)
        label_dialog = "KI67";
    else if (currentStaining == 7)
        label_dialog = "ECAD";
    else if (currentStaining == 8)
        label_dialog = "CK19";
    else if (currentStaining == 9)
        label_dialog = "HER";
    return label_dialog;

}


/*** Forzar zoom 100%
 */
var scale = 'scale(1)';
document.body.style.webkitTransform = scale;    // Chrome, Opera, Safari
document.body.style.msTransform = scale;       // IE 9
document.body.style.transform = scale;     // General


/*** Deshabilitar boton derecho
 */

document.addEventListener("contextmenu", function (e) {
    e.preventDefault();
}, false);


/***0
 Label handling functions
 */


function getLabelNameById(id) {
    for (var i = 0; i < labelDictionary.length; i++) {
        if (labelDictionary[i].uid == id) {
            return labelDictionary[i].label;
        }
    }
    return -1;
}


function getLabelIdByName(name) {
    for (var i = 0; i < labelDictionary.length; i++) {
        if (labelDictionary[i].label == name) {
            return labelDictionary[i].uid;
        }
    }
    return -1;
}

function getLabelColorById(id) {
    for (var i = 0; i < labelDictionary.length; i++) {
        if (labelDictionary[i].uid == id) {
            return labelDictionary[i].color;
        }
    }
    return -1;
}

function getLabelColorByName(name) {
    for (var i = 0; i < labelDictionary.length; i++) {
        if (labelDictionary[i].label == name) {
            return labelDictionary[i].color;
        }
    }
    return -1;
}

function getLabelAlphaColorByName(name) {
    for (var i = 0; i < labelDictionary.length; i++) {
        if (labelDictionary[i].label == name) {
            return labelDictionary[i].alpha;
        }
    }
    return -1;
}

function appendRegionsToList() {
    label = currentLabel.label;

    $('#regionBar').css('background-color', 'rgba(' + currentLabel.color.red + ',' + currentLabel.color.green + ',' + currentLabel.color.blue + ',' + 0.9 + ')');
    $('#regionBar').css('top', document.getElementById('regionList').offsetTop);
    $('#regionBar').show();
    var subRegionListNode = document.getElementById("subRegionList");
    subRegionListNode.innerHTML = '';
    var scoreListNode = document.getElementById("scoreList");
    scoreListNode.innerHTML = '';
    for (var i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
        if (ImageInfo[currentImage]["Regions"][i].name == label) {
            var c = getLabelColorByName(label);
            var str = ["<div class='subregion-tag' id='r" + String(ImageInfo[currentImage]["Regions"][i].uid) + "'> " +
            "<span style='padding-left: 7px; padding-right: 7px'>"
            + "<span style='weight: bold; color:white'>"
            + String.fromCodePoint(0x26F6) + "</span>"
            + " " +
            "<span style='weight: bold; background-color:white; padding-left: 5px;padding-right: 5px; text-align:center'>" +
            +String(ImageInfo[currentImage]["Regions"][i].uid) + "</span></span></div>"
            ].join(" ");
            var el = $(str);
            $("#subRegionList").append(el);
            el.click(singleClickOnRegion);

        }
    }
    //appendScoreToRegion();
}

function appendScoreToRegion(i_selected) {

    label = currentLabel.label;

    var ofl = parseInt(document.getElementById('regionBar').offsetLeft);
    var ofw = parseInt(document.getElementById('regionBar').offsetWidth);

    $('#scoreBar').css('background-color', 'rgba(255, 255, 255, 0)');
    $('#scoreBar').css('top', document.getElementById('regionList').offsetTop);
    $('#scoreBar').css('left', String(ofl + ofw) + "px");



    var scoreListNode = document.getElementById("scoreList");
    scoreListNode.innerHTML = '';

    if (label == "ROI_Tumoral" && currentStaining >= 2) {
        if (currentStaining < 7) {
            for (var i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
                if (ImageInfo[currentImage]["Regions"][i].name == label) {
                    var texto;
                    var staining_name = getNameFromCurrentStaining();
                    if (typeof ImageInfo[currentImage]["Regions"][i].score !== "undefined") {
                        texto = staining_name + ": " + String((ImageInfo[currentImage]["Regions"][i].score * 100).toFixed(2)) + " %";
                    } else {
                        texto = staining_name + ": -"
                    }
                    var str = ["<div class='subscore-tag' id='s" + String(ImageInfo[currentImage]["Regions"][i].uid) + "'> " +
                    "<span style='weight: bold; background-color:white; padding-left: 5px;padding-right: 5px; text-align:center'>"
                    + texto + "</span></span></div>"].join(" ");
                    var el = $(str);
                    $("#scoreList").append(el);
                }
            }
        }
        else if (currentStaining == 9 || currentStaining == 7) {


            var texto;
            if (typeof ImageInfo[currentImage]["Regions"][i_selected].score !== "undefined") {
                var score = JSON.parse("[" + ImageInfo[currentImage]["Regions"][i_selected].score + "]")[0];
                var total = 0;
                for (var is = 0; is < score.length-1; is++) {
                    total = total + score[is];
                }
                if (total==0)
                {
                    total==1;
                }

                var str = ["<div class='subscore-tag' id='s" + String(ImageInfo[currentImage]["Regions"][i_selected].uid) + "'> " +
                    "<table style='width:100%border-spacing: 0;border-collapse: collapse;'>"+ "<tr> <th><span style='display: inline-block; width: 4em;weight: bold; background-color:white; text-align:center'>"+ "HER2" + "</span></th>"+
                    "<th><span style='display: inline-block; width: 4em;weight: bold; background-color:white; text-align:center'>"+ score[4] + "</span></th></tr></table></div>"].join(" ");

                var el = $(str);

                $("#scoreList").append(el);

                var str = ["<div class='subscore-tag' id='s" + String(ImageInfo[currentImage]["Regions"][i_selected].uid) + "0" + "'> " +
                    "<table style='width:100%border-spacing: 0;border-collapse: collapse;'>"+ "<tr> <th><span style='display: inline-block; width: 4em;color:black;weight: bold; background-color: rgb(255,250,66);text-align:center'>"+ "0" + "</span></th>"+
                    "<th><span style='display: inline-block; width: 4em;color:black; background-color: rgb(255,250,66);  text-align:center'>"+ String(score[0]) + "</span></th>" +
                    "<th><span style='display: inline-block; width: 8em;color:black;weight: bold; background-color: rgb(255,250,66);  text-align:center'>"+ String((score[0]/total * 100).toFixed(2)) + " %" + "</span></th></tr></table></div>"].join(" ");

                var el = $(str);
                $("#scoreList").append(el);

                 var str = ["<div class='subscore-tag' id='s" + String(ImageInfo[currentImage]["Regions"][i_selected].uid) + "1+" + "'> " +
                    "<table style='width:100%border-spacing: 0;border-collapse: collapse;'>"+ "<tr> <th><span style='display: inline-block; width: 4em;color:white;weight: bold; background-color: rgb(66,183,185);text-align:center'>"+ "1+" + "</span></th>"+
                    "<th><span style='display: inline-block; width: 4em;color:white; background-color: rgb(66,183,185);  text-align:center'>"+ String(score[1]) + "</span></th>" +
                    "<th><span style='display: inline-block; width: 8em;color:white;weight: bold; background-color: rgb(66,183,185);  text-align:center'>"+ String((score[1]/total * 100).toFixed(2)) + " %" + "</span></th></tr></table></div>"].join(" ");
                var el = $(str);
                $("#scoreList").append(el);

                var str = ["<div class='subscore-tag' id='s" + String(ImageInfo[currentImage]["Regions"][i_selected].uid) + "2+" + "'> " +
                    "<table style='width:100%border-spacing: 0;border-collapse: collapse;'>"+ "<tr> <th><span style='display: inline-block; width: 4em;color:black;weight: bold; background-color: rgb(214,145,193);text-align:center'>"+ "2+" + "</span></th>"+
                    "<th><span style='display: inline-block; width: 4em;color:black; background-color: rgb(214,145,193);  text-align:center'>"+ String(score[2]) + "</span></th>" +
                    "<th><span style='display: inline-block; width: 8em;color:black;weight: bold; background-color: rgb(214,145,193);  text-align:center'>"+ String((score[2]/total * 100).toFixed(2)) + " %" + "</span></th></tr></table></div>"].join(" ");
                var el = $(str);
                $("#scoreList").append(el);

                var str = ["<div class='subscore-tag' id='s" + String(ImageInfo[currentImage]["Regions"][i_selected].uid) + "3+" + "'> " +
                    "<table style='width:100%border-spacing: 0;border-collapse: collapse;'>"+ "<tr> <th><span style='display: inline-block; width: 4em;color:white;weight: bold; background-color: rgb(199,93,171);text-align:center'>"+ "3+" + "</span></th>"+
                    "<th><span style='display: inline-block; width: 4em;color:white; background-color: rgb(199,93,171);  text-align:center'>"+ String(score[3]) + "</span></th>" +
                    "<th><span style='display: inline-block; width: 8em;color:white;weight: bold; background-color: rgb(199,93,171);  text-align:center'>"+ String((score[3]/total * 100).toFixed(2)) + " %" + "</span></th></tr></table></div>"].join(" ");

                var el = $(str);
                $("#scoreList").append(el);
            }

            //el.click(singleClickOnRegion);


        }


    }
}


function appendLabelToList(label) {
    var str = ["<div class='region-tag' id='" + label.uid + "'>",
        "<img class='eye' title='Region visible' id='eye_" + label.uid + "' src='" + staticPath + "/img/eyeOpened.svg' />",
        "<div class='region-color' id='color_" + label.uid + "'",
        "style='background-color:rgba(",
        parseInt(label.color.red), ",", parseInt(label.color.green), ",", parseInt(label.color.blue), ",", label.alpha,
        ")'></div>",
        "<span class='region-name' id='name_" + label.uid + "' style='overflow:hidden'>" + label.label +
        '</span> (<span id="count_' + label.uid + '">0</span>)<br/>',
    ].join(" ");

    var el = $(str);
    $("#regionList").append(el);

    // handle single click on computers
    el.click(singleClickOnLabel);
    // select latest label
    var tags = $("#regionList > .region-tag");
    selectLabel(tags[tags.length - 1]);

    // create count entry
    labelIds[label.label] = label.uid;
    countLabel[label.label] = 0;
}


function appendLabelsToList() {
    var tags = $("#regionList > .region-tag");
    tags.each(function (i) {
        tags[i].remove();
    });

    if (labelDictionary) {
        for (var i = 0; i < labelDictionary.length; i++) {
            appendLabelToList(labelDictionary[i]);
        }
        // select first label automatically
        selectNextLabel();
    }
}

function selectNextLabel() {
    var index = 0;
    var labelTags = $("#regionList > .region-tag");
    if (currentLabel.uid) {
        for (var i = 0; i < labelTags.length; i++) {
            if (labelTags[i].id == currentLabel.uid && i + 1 < labelTags.length) {
                index = i + 1;
                break;
            }
        }
    }
    selectLabel(labelTags[index]);
}

function selectNextRegionLabel() {
    var index = 0;
    var labelRegionTags = $("#subregionList > .subregion-tag");


    for (var i = 0; i < labelRegionTags.length; i++) {
        if (labelRegionTags[i].classList.contains("selected") && i + 1 < labelRegionTags.length) {
            index = i + 1;
            break;
        }
    }

     for (var i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
        if (labelRegionTags[index].id == ["r" + ImageInfo[currentImage]["Regions"][i].uid]) {
            var coords = ImageInfo[currentImage]["Regions"][i].path.getFirstSegment().point;
            var imgCoords = convertPathToImgCoordinates(coords);
            var imgPoint = new OpenSeadragon.Point(imgCoords.x, imgCoords.y);
            var vPoint = viewer.viewport.imageToViewportCoordinates(imgPoint);
            viewer.viewport.panTo(new OpenSeadragon.Point(vPoint.x, vPoint.y));
            viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(ImageInfo[currentImage]["Regions"][i].zoom));
            selectRegion(ImageInfo[currentImage]["Regions"][i]);
            i_selected = i;
            break;
        }
    }

    appendScoreToRegion(i_selected);


    selectRegionLabel(labelRegionTags[index]);
}

function selectLabel(el) {
    for (var i = 0; i < labelDictionary.length; i++) {
        if (el.id == labelDictionary[i].uid) {
            currentLabel = labelDictionary[i];
            $("#regionList > .region-tag").each(function (i) {
                $(this).addClass("deselected");
                $(this).removeClass("selected");
            });
            var tag = $("#regionList > .region-tag#" + currentLabel.uid);
            $(tag).removeClass("deselected");
            $(tag).addClass("selected");
            break;
        }
    }
}

function selectRegionLabel(el) {
    for (var i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
        if (el.id == ["r" + ImageInfo[currentImage]["Regions"][i].uid]) {

            $("#subregionList > .subregion-tag").each(function (i) {
                $(this).addClass("deselected");
                $(this).removeClass("selected");
            });
            var tag = $("#subregionList > .subregion-tag#" + el.id);
            $(tag).removeClass("deselected");
            $(tag).addClass("selected");
            break;
        }
    }
}

/***1
 Region handling functions
 */

function newRegion(arg, imageNumber) {
    var reg = {};

    if (arg.uid) {
        reg.uid = arg.uid;
    } else {
        reg.uid = uniqueID();
    }

    if (arg.score) {
        reg.score = arg.score;
    } else {
        reg.score = undefined;
    }


    if (arg.name) {
        reg.name = arg.name;
    } else {
        reg.name = currentLabel.label;
    }
    if (arg.imgCoords) {
        reg.imgCoords = arg.imgCoords
    }
    if (arg.color) {
        color = arg.color;
    } else {
        color = currentLabel.color ? currentLabel.color : regionHashColor(reg.name);
    }
    if (arg.path) {
        reg.path = arg.path;
        reg.path.strokeWidth = arg.path.strokeWidth ? arg.path.strokeWidth : config.defaultStrokeWidth;
        reg.path.strokeColor = arg.path.strokeColor ? arg.path.strokeColor : config.defaultStrokeColor;
        reg.path.strokeScaling = false;

        reg.path.fillColor = arg.path.fillColor ? arg.path.fillColor : 'rgba(' + color.red + ',' + color.green + ',' + color.blue + ',' + config.defaultFillAlpha + ')';
        reg.path.selected = false;
    }

    if (arg.zoom) {
        reg.zoom = arg.zoom;
    }


    if (imageNumber === undefined) {
        imageNumber = currentImage;
    }

    // push the new region to the Regions array
    ImageInfo[imageNumber]["Regions"].push(reg);
    // increase region count
    countAll++;
    countLabel[reg.name] += 1;
    $('#count_all').html(countAll);
    $('#count_' + labelIds[reg.name]).html(countLabel[reg.name]);
    return reg;
}

function removeRegion(reg, imageNumber) {
    if (debug) console.log("> removeRegion");

    if (imageNumber === undefined) {
        imageNumber = currentImage;
    }

    // remove from Regions array
    ImageInfo[imageNumber]["Regions"].splice(ImageInfo[imageNumber]["Regions"].indexOf(reg), 1);
    if (reg.path) {
        // remove from paths
        reg.path.remove();
    }

    // lower region count
    countAll--;
    countLabel[reg.name] -= 1;
    $('#count_' + labelIds[reg.name]).html(countLabel[reg.name]);
    $('#count_all').html(countAll);
}

function selectRegion(reg) {
    if (debug) console.log("> selectRegion");

    var i, i_selected;

    for (i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
        if (ImageInfo[currentImage]["Regions"][i] == reg) {
            // Select region
            region = reg;
            if (ImageInfo[currentImage]["Regions"][i].path) {
                // Select path
                i_selected = i;
                reg.path.selected = true;

            }
        } else {
            if (ImageInfo[currentImage]["Regions"][i].path) {
                // Deselect path
                ImageInfo[currentImage]["Regions"][i].path.selected = false;
            }
        }
    }
    //Actualización de panel de etiquetas
    var labelTags = $("#regionList > .region-tag");
    for (var i = 0; i < labelTags.length; i++) {
        if (labelTags[i].id == getLabelIdByName(reg.name)) {
            selectLabel(labelTags[i]);
            break;
        }
    }
    appendRegionsToList();

    $("#subregionList > .subregion-tag").each(function () {
        $(this).addClass("deselected");
        $(this).removeClass("selected");
    });

    var tag = $("#subregionList > .subregion-tag#" + "r" + reg.uid);
    $(tag).removeClass("deselected");
    $(tag).addClass("selected");

    paper.view.draw();
    appendScoreToRegion(i_selected);
}

function deselectRegion(reg) {
    if (reg) {
        if (region.uid == reg.uid) {
            region = null;
        }
        for (var i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
            if (ImageInfo[currentImage]["Regions"][i] == reg) {
                // Deselect path
                ImageInfo[currentImage]["Regions"][i].path.selected = false;
                //ImageInfo[currentImage]["Regions"][i].path.fullySelected = false;
                paper.view.draw();
            }
        }
    }
}

function findRegionByUID(uid) {
    if (debug) console.log("> findRegionByUID");
    var i;
    if (debug > 2) console.log("look for uid: " + uid);
    if (debug > 2) console.log("region array lenght: " + ImageInfo[currentImage]["Regions"].length);

    for (i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {

        if (ImageInfo[currentImage]["Regions"][i].uid == uid) {
            if (debug > 2) console.log("region " + ImageInfo[currentImage]["Regions"][i].uid + ": ");
            if (debug > 2) console.log(ImageInfo[currentImage]["Regions"][i]);
            return ImageInfo[currentImage]["Regions"][i];
        }
    }
    console.log("Region with unique ID " + uid + " not found");
    return null;
}

function findRegionByName(name) {
    if (debug) console.log("> findRegionByName");
    var i;
    for (i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
        if (ImageInfo[currentImage]["Regions"][i].name == name) {
            return ImageInfo[currentImage]["Regions"][i];
        }
    }
    console.log("Region with name " + name + " not found");
    return null;
}

function uniqueID(forLabelDictionary) {
    if (forLabelDictionary) {
        if (debug) console.log("> uniqueID");
        return labelDictionary.length > 0 ? parseInt(labelDictionary[labelDictionary.length - 1].uid) + 1 : 1;
    } else {
        var regions = ImageInfo[currentImage].Regions;
        return regions.length > 0 ? regions[regions.length - 1].uid + 1 : 1;
    }
    return 0;
}

function hash(str) {
    var result = str.split("").reduce(function (a, b) {
        a = ((a << 5) - a) + b.charCodeAt(0);
        return a & a;
    }, 0);
    return result;
}

function regionHashColor(name) {

    var color = {};
    var h = hash(name);

    // add some randomness
    h = Math.sin(h++) * 10000;
    h = 0xffffff * (h - Math.floor(h));

    color.red = h & 0xff;
    color.green = (h & 0xff00) >> 8;
    color.blue = (h & 0xff0000) >> 16;
    return color;
}

function regionPicker(parent) {
    if (debug) console.log("> regionPicker");

    $("div#regionPicker").appendTo("body");
    $("div#regionPicker").show();
}

function changeRegionName(reg, name) {
    if (debug) console.log("> changeRegionName");

    var color = regionHashColor(name);

    if (reg.path) {
        // Update path
        reg.name = name;
        reg.path.fillColor = 'rgba(' + color.red + ',' + color.green + ',' + color.blue + ',0.5)';
        paper.view.draw();
    }

    // Update region tag
    $(".region-tag#" + reg.uid + ">.region-name").text(name);
    $(".region-tag#" + reg.uid + ">.region-color").css('background-color', 'rgba(' + color.red + ',' + color.green + ',' + color.blue + ',0.67)');
}

function sendRegionsToBack(name) {


    for (var i in ImageInfo[currentImage]["Regions"]) {
        var reg = ImageInfo[currentImage]["Regions"][i];

        if (reg.name.toLowerCase().includes(name)) {
            reg.path.sendToBack();
        }

    }
}

/*** Toggle visibility of region
 ***/

function toggleAllRegions() {
    var toggleEye = $('#toggle-poi');
    if (toggleEye[0].src.indexOf("eyeOpened.svg") === -1) {
        toggleEye.attr('src', staticPath + '/img/eyeOpened.svg');
    } else {
        toggleEye.attr('src', staticPath + '/img/eyeClosed.svg');
    }
    for (var i = 0; i < labelDictionary.length; i++) {
        var eye = $('#eye_' + labelDictionary[i].uid);
        if (eye[0].src != toggleEye[0].src)
            toggleRegions(labelDictionary[i].label);
    }
}

function toggleRegions(name) {
    if (debug) console.log("< toggle region");
    var regions = ImageInfo[0].Regions;
    for (var i = 0; i < regions.length; i++) {
        var reg = regions[i];
        if (reg.name == name) {
            if (reg.path.fillColor !== null) {
                reg.path.storeColor = reg.path.fillColor;
                reg.path.fillColor = null;

                reg.path.storeWidth = reg.path.strokeWidth;
                reg.path.strokeWidth = 0;
                //reg.path.fullySelected = false;
            } else {
                reg.path.fillColor = reg.path.storeColor;
                reg.path.strokeWidth = reg.path.storeWidth;
            }
        }
    }
    paper.view.draw();
    var eye = $('#eye_' + labelIds[name]);
    if (eye[0].src.indexOf("eyeOpened.svg") === -1) {
        eye.attr('src', staticPath + '/img/eyeOpened.svg');
    } else {
        eye.attr('src', staticPath + '/img/eyeClosed.svg');
        if (region) {
            if (region.name == name) {
                deselectRegion(region);
            }
        }
    }
    if (debug) console.log("> toggle region");
}

function changeRegionAnnotationStyle(uid) {
    if (debug) console.log("< changeRegionAnnotationStyle");
    for (var i = 0; i < labelDictionary.length; i++) {
        if (labelDictionary[i].uid == uid) {
            annotationStyle(labelDictionary[i]);
            break;
        }
    }
    if (debug) console.log("> changeRegionAnnotationStyle");
}

/***2
 Interaction: mouse and tap
 */
function convertPathToImgCoordinates(point) {
    // convert to screen coordinates
    var screenCoords = paper.view.projectToView(point);
    // convert to viewport coordinates
    var viewportCoords = viewer.viewport.pointFromPixel(new OpenSeadragon.Point(screenCoords.x, screenCoords.y));
    // convert to image coordinates
    var imgCoords = viewer.viewport.viewportToImageCoordinates(viewportCoords);
    return imgCoords;
}

function convertImgToPathCoordinates(point) {
    // convert to viewport coordinates
    var viewportCoords = viewer.viewport.imageToViewportCoordinates(point);
    // convert to screen coordinates
    var pixel = viewer.viewport.pixelFromPoint(viewportCoords);
    // convert to project coordinates
    var projectCoords = paper.view.viewToProject(pixel);
    return projectCoords;
}

function clickHandler(event) {
    if (debug) {
        console.log("> clickHandler");
        var webPoint = event.position;
        var viewportPoint = viewer.viewport.pointFromPixel(webPoint);
    }
    event.stopHandlers = !navEnabled;

    if (selectedTool == "roib") {
        addRoi(event, 'N');
    } else if (selectedTool == "roim") {
        addRoi(event, 'T');
    }
    if ((selectedTool == "roib" || selectedTool == "roim") && buttonMode == false) {
        clearToolSelection();
        navEnabled = true;
        paper.view.draw();
    }
}

function releaseHandler(event) {
    if (debug) {
        console.log("> releaseHandler");
    }
    if (buttonMode == true) {
        mouseUp();
        if (selectedTool == "draw") {
            $('body').css('cursor', 'url(/static/cursors/editinibica_cursor.svg),auto');
            selectToolOnKeyPress("draw");
            buttonMode = true;
        }
        if (selectedTool == "roim" || selectedTool == "roib" || selectedTool == "selected") {
            buttonMode = true;
        }
    }

}

function getZoom() {
    var zoom = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true));
    zoom = Math.round(zoom * 10000) / 10000;
    return zoom;
}

function addRoi(event, roitype) {

    console.log(selectedTool);
    var webPoint = event.position;
    var point = paper.view.viewToProject(new paper.Point(webPoint.x, webPoint.y));
    var viewportPoint = viewer.viewport.pointFromPixel(webPoint);
    var imgCoord = viewer.viewport.viewportToImageCoordinates(viewportPoint);
    var x = imgCoord.x;
    var y = imgCoord.y;

    if (currentStaining >1) {
        w_roi_size = 700;
    }
    else {
        w_roi_size = 500;
    }
    x_roi = [-w_roi_size, w_roi_size, w_roi_size, -w_roi_size];
    y_roi = [-w_roi_size, -w_roi_size, w_roi_size, w_roi_size];

    var path = new paper.Path();

    var imgPoints = [];
    for (var i = 0; i < x_roi.length; i++) {
        var imgPoint = new OpenSeadragon.Point(x + x_roi[i], y + y_roi[i]);
        var point = convertImgToPathCoordinates(imgPoint);
        path.add(point);
        imgPoints.push(imgPoint)
    }
    path.closed = true;
    if (roitype == 'N') {
        var regionColor = getLabelColorById(1);
        path.fillColor = new paper.Color(regionColor.red / 255.0, regionColor.green / 255.0, regionColor.blue / 255.0);
        path.strokeColor = path.fillColor;
        path.fillColor.alpha = 0;
        path.closed = true;
        path.strokeWidth = 5;
        path.sendToBack();
        region = newRegion({path: path, imgCoords: imgPoints, name: 'ROI_NonTumoral', zoom: getZoom()});
        sendRegionsToBack('colormap');

    } else if (roitype == 'T') {
        var regionColor = getLabelColorById(2);
        path.fillColor = new paper.Color(regionColor.red / 255.0, regionColor.green / 255.0, regionColor.blue / 255.0);
        path.strokeColor = path.fillColor;
        path.fillColor.alpha = 0;
        path.closed = true;
        path.strokeWidth = 5;
        path.sendToBack();
        region = newRegion({path: path, imgCoords: imgPoints, name: 'ROI_Tumoral', zoom: getZoom()});
        sendRegionsToBack('colormap');
    }
    navEnabled = true;
    paper.view.draw();
    selectRegion(region);

}

function selectAnnotationToImport(event) {
    deselectAllButtons();
    console.log("> Select annotation");
    navEnabled = false;
    //Check annotation files
    $.ajax({
        type: "GET",
        url: "/annotation/getAnnotationFiles?src=" + slideName,
    }).done(function (annotation_files) {
        // Custom DOM/jQuery Element
        var select = $("<select>", {
            css: {
                "width": "100%",
                "margin-top": "1rem"
            }
        });
        console.log(annotation_files);
        for (var i = 0; i < annotation_files.length; i++) {
            select.append("<option>" + annotation_files[i] + "</option>");
        }
        $.MessageBox({
            message: "Please, select an annotation file:",
            input: select,
            buttonDone: "OK",
            buttonFail: "Cancel",
            filterFail: function () {
                navEnabled = true;
            }
        }).done(function (data) {
            if (!$.isEmptyObject(data)) {
                $.LoadingOverlay("show");
                importAnnotation(data);
            } else {
                navEnabled = true;
            }

        });
    });
}

function addImportedRegions(json, selected_name, selected_zoom, simplify) {

    var regions = JSON.parse(json);
    for (var i = 0; i < regions.length; i++) {
        deleted = false;
        var reg = regions[i];
        if (selected_name != "") {
            reg.name = selected_name;
        }
        if (selected_zoom != "") {
            reg.zoom = selected_zoom;
        }
        //console.log(regions)
        if (reg.path[0] == "CompoundPath") {
            for (var npath = 0; npath < reg.imgCoords.length; npath++) {
                if (reg.imgCoords[npath].length > 0) {
                    var path = new paper.Path();
                    path.strokeWidth = config.defaultStrokeWidth;
                    var imgPoints = [];
                    for (var j = 0; j < reg.imgCoords[npath].length; j++) {
                        var imgPoint = new OpenSeadragon.Point(reg.imgCoords[npath][j].x, reg.imgCoords[npath][j].y);
                        var point = convertImgToPathCoordinates(imgPoint);
                        path.add(point);
                        imgPoints.push(imgPoint);
                    }
                    path.closed = true;
                    if (simplify) {
                        path.simplify(0.0005);
                    }
                    regionColor = getLabelColorByName(reg.name);
                    path.fillColor = new paper.Color(regionColor.red / 255.0, regionColor.green / 255.0, regionColor.blue / 255.0);
                    path.fillColor.alpha = 0.5;
                    if (getLabelIdByName(reg.name) <= 2) {
                        path.strokeColor = path.fillColor;
                        path.strokeColor.alpha = 1;
                        path.fillColor.alpha = 0;
                        path.closed = true;
                        path.strokeWidth = 5;
                    }
                    region = newRegion({path: path, imgCoords: imgPoints, name: reg.name, zoom: getZoom()});
                }
            }

        } else if (reg.path[0] == "Path") {

            if (reg.imgCoords.length > 0) {
                var path = new paper.Path();
                path.strokeWidth = config.defaultStrokeWidth;
                var imgPoints = [];
                for (var j = 0; j < reg.imgCoords.length; j++) {
                    var imgPoint = new OpenSeadragon.Point(reg.imgCoords[j].x, reg.imgCoords[j].y);
                    var point = convertImgToPathCoordinates(imgPoint);
                    path.add(point);
                    imgPoints.push(imgPoint);
                }
                if (Math.abs(path.area) < 0.03) {
                    //path.remove();
                    //deleted = true;

                }
                //else{ CAMBIAR
                if (1) {

                    path.closed = true;
                    regionColor = getLabelColorByName(reg.name);
                    path.fillColor = new paper.Color(regionColor.red / 255.0, regionColor.green / 255.0, regionColor.blue / 255.0);
                    //console.log(reg);
                    //console.log(path);
                    if ((reg.path[1]).hasOwnProperty("fillColor")) {
                        path.fillColor.alpha = reg.path[1].fillColor[3];
                    }else{
                        path.fillColor.alpha = 0.5;
                    }

                    if ((reg.path[1]).hasOwnProperty("strokeAlpha")) {
                        path.strokeColor = new paper.Color(reg.path[1].strokeColor[0] / 255.0, reg.path[1].strokeColor[1] / 255.0, reg.path[1].strokeColor[2] / 255.0);
                        path.strokeColor.alpha = reg.path[1].strokeAlpha;
                    } else if (reg.path[1].hasOwnProperty("strokeColor")) {
                        if (reg.path[1].strokeColor.length == 4) {
                            path.strokeColor = new paper.Color(reg.path[1].strokeColor[0] / 255.0, reg.path[1].strokeColor[1] / 255.0, reg.path[1].strokeColor[2] / 255.0);
                            path.strokeColor.alpha = reg.path[1].strokeColor[3];
                        }
                    }

                    //ROI TUMORAL Y NO TUMORAL
                    if (getLabelIdByName(reg.name) <= 2) {
                        path.strokeColor = path.fillColor;
                        path.strokeColor.alpha = 1;
                        path.fillColor.alpha = 0;
                        path.closed = true;
                        path.strokeWidth = 5;
                    }
                    if (!reg.hasOwnProperty("score"))
                        reg.score = undefined;
                    region = newRegion({
                        path: path,
                        imgCoords: imgPoints,
                        name: reg.name,
                        zoom: reg.zoom,
                        score: reg.score
                    });
                }
            }
        }
        if (deleted == false) {
            /*
            var holes = false;
            //do {
                holes = false;
                var area = [];
                for (var j = 0; j < ImageInfo[currentImage]["Regions"].length; j++) {
                    area[j] = ImageInfo[currentImage]["Regions"][j].path.area;
                }
                var indices = Array.from(Array(area.length).keys())
                    .sort((a, b) => area[a] < area[b] ? -1 : (area[b] < area[a]) | 0);

                for (var j = 0; j < ImageInfo[currentImage]["Regions"].length; j++) {
                    region = ImageInfo[currentImage]["Regions"][indices[j]];

                    if (!(region.path instanceof paper.CompoundPath)) {
                        var old_index = getIndexFromPath(region.path);
                        var new_index = solvePathHoles(old_index);
                        if (new_index != old_index) {
                            holes = true;
                            break;
                        }
                    }
                }
            //} while (holes == true);
            */
        }
        paper.view.draw();
    }

}

function updateRegionImgCoords(region_index) {
    ImageInfo[currentImage]["Regions"][region_index].imgCoords.splice(0, ImageInfo[currentImage]["Regions"][region_index].imgCoords.length);
    //ImageInfo[currentImage]["Regions"][region_index].zoom = getZoom();
    if (ImageInfo[currentImage]["Regions"][region_index].path instanceof paper.CompoundPath) {
        for (var j = 0; j < ImageInfo[currentImage]["Regions"][region_index].path.children.length; j++) {
            var children = [];
            for (var np = 0; np < ImageInfo[currentImage]["Regions"][region_index].path.children[j]._segments.length; np++) {
                children.push(
                    convertPathToImgCoordinates(ImageInfo[currentImage]["Regions"][region_index].path.children[j]._segments[np]._point));
            }
            ImageInfo[currentImage]["Regions"][region_index].imgCoords.push(children);
        }
    } else {
        for (var np = 0; np < ImageInfo[currentImage]["Regions"][region_index].path._segments.length; np++) {
            ImageInfo[currentImage]["Regions"][region_index].imgCoords.push(
                convertPathToImgCoordinates(ImageInfo[currentImage]["Regions"][region_index].path._segments[np]._point));
        }
    }
}

function importAnnotation(strAnnotation) {
    console.log("> Import annotation");
    var posExt = strAnnotation.lastIndexOf(".");
    var annotationFormat = strAnnotation.substring(posExt + 1,);


    if (annotationFormat.match(/xml/i)) {
        $.ajax({
            type: "GET",
            url: "/annotation/importAnnotationXML?src=" + strAnnotation,
        }).done(function (o) {
            if (o == "404") {
                clearToolSelection();

            } else {

                currentAnnotation = strAnnotation;
                $('#currentAnnotationName').html(currentAnnotation);
                while (ImageInfo[currentImage].Regions.length > 0) {
                    removeRegion(ImageInfo[currentImage].Regions[0]);
                }

                var annotation = o;
                var imgCoords = annotation.coords;
                var labels = annotation.labels;
                var texts = annotation.texts;
                var zooms = annotation.zooms;

                for (var i = 0; i < imgCoords.length; i++) {
                    var path = new paper.Path();
                    path.strokeWidth = config.defaultStrokeWidth;
                    var imgPoints = [];
                    for (var j = 0; j < imgCoords[i].length; j++) {
                        var imgPoint = new OpenSeadragon.Point(imgCoords[i][j][0], imgCoords[i][j][1]);
                        var point = convertImgToPathCoordinates(imgPoint);
                        path.add(point);

                        imgPoints.push(imgPoint)
                    }
                    path.closed = true;
                    regionColor = getLabelColorById(labels[i]);
                    path.fillColor = new paper.Color(regionColor.red / 255.0, regionColor.green / 255.0, regionColor.blue / 255.0);
                    path.fillColor.alpha = 0.5;
                    if (labels[i] > 2) {
                        path.simplify(0.0005);
                    } else {
                        path.strokeColor = path.fillColor;
                        path.strokeColor.alpha = 1;
                        path.fillColor.alpha = 0;
                        path.closed = true;
                        path.strokeWidth = 5;
                    }
                    region = newRegion({path: path, imgCoords: imgPoints, name: texts[i], zoom: zooms[i]});
                    paper.view.draw();
                    selectRegion(region);
                }
                appendRegionsToList();

                $.LoadingOverlay("hide");
            }
        });

    } else if (annotationFormat.match(/json/i)) {
        $.ajax({
            type: "GET",
            url: "/annotation/loadJson?src=" + strAnnotation,
        }).done(function (json) {
            if (json == "404") {
                clearToolSelection();

            } else {
                currentAnnotation = strAnnotation;
                $('#currentAnnotationName').html(currentAnnotation);
                while (ImageInfo[currentImage].Regions.length > 0) {
                    removeRegion(ImageInfo[currentImage].Regions[0]);
                }
                toggleAllRegions();
                toggleAllRegions();
                addImportedRegions(json, "", "", false);
                $.LoadingOverlay("hide");
                setRegionDepths();
                appendRegionsToList();

            }


        });
    } else {
        $.LoadingOverlay("hide");
    }

    navEnabled = true;
    backToSelect();
}

function pressHandler(event) {
    if (debug) console.log("> pressHandler");

    var dictListContent = $('#dicts_content');
    if (dictListContent.is(":visible")) {
        dictListContent.hide()
    }

    if (!navEnabled) {
        event.stopHandlers = true;
        mouseDown(event.originalEvent.layerX, event.originalEvent.layerY);
    }
    resizeAnnotationOverlay();
}

function dragHandler(event) {
    if (debug > 1)
        console.log("> dragHandler");

    if (!navEnabled) {
        event.stopHandlers = true;
        mouseDrag(event.originalEvent.layerX, event.originalEvent.layerY, event.delta.x, event.delta.y);
    }
}

function dragEndHandler(event) {
    if (debug) console.log("> dragEndHandler");
    if (!navEnabled) {
        event.stopHandlers = true;
    }
}


var contextFlag = false;

function singleClickOnLabel(event) {
    if (debug) console.log("> labelClick");
    event.stopPropagation();
    event.preventDefault();
    var clickedId = event.toElement.id;
    var el = $(this);

    if (clickedId === "eye_" + el[0].id) {
        // toogle visibility
        toggleRegions(getLabelNameById(el[0].id));
    } else if (clickedId === "color_" + el[0].id) {
        //changeRegionAnnotationStyle(el[0].id);
    } else {
        selectLabel(el[0]);
    }

    appendRegionsToList();

    if (debug) console.log("< labelClick");
}

function singleClickOnRegion(event) {
    if (debug) console.log("> regionClick");
    event.stopPropagation();
    event.preventDefault();
    var el = $(this);
    var i_selected;
    for (var i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
        if (el[0].id == ["r" + ImageInfo[currentImage]["Regions"][i].uid]) {
            var coords = ImageInfo[currentImage]["Regions"][i].path.getFirstSegment().point;
            var imgCoords = convertPathToImgCoordinates(coords);
            var imgPoint = new OpenSeadragon.Point(imgCoords.x, imgCoords.y);
            var vPoint = viewer.viewport.imageToViewportCoordinates(imgPoint);
            viewer.viewport.panTo(new OpenSeadragon.Point(vPoint.x, vPoint.y));
            viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(ImageInfo[currentImage]["Regions"][i].zoom));
            selectRegion(ImageInfo[currentImage]["Regions"][i]);
            i_selected = i;

            break;
        }
    }
    if (debug) console.log("< regionClick");
    selectRegionLabel(el[0]);
    appendScoreToRegion(i_selected);
}


function singlePressOnRegion(event) {
    if (debug) console.log("> singlePressOnRegion");
    if (debug) console.log(event);

    event.stopPropagation();
    event.preventDefault();

    var el = $(this);
    var uid;
    var reg;
    var clickedId = event.toElement.id;


    uid = $(this).attr('id');

    if (clickedId === "eye_" + uid) {
        // toogle visibility
        toggleRegions(getLabelNameById(this.id));
    } else if (clickedId === "name_" + uid) {
        // Click on regionList (list or annotated regions)
        reg = findRegionByUID(uid);
        if (reg) {
            selectRegion(reg);
        } else {
            console.log("region undefined");
        }
    } else if (clickedId === "color_" + uid) {
        // open color picker
        var reg = findRegionByUID(this.id);
        if (reg.path.fillColor != null) {
            if (reg) {
                selectRegion(reg);
            }
            annotationStyle(reg);
        }
    } else {
        // Click on regionList (list or annotated regions)
        reg = findRegionByUID(uid);
        if (reg) {
            selectRegion(reg);
        } else {
            console.log("region undefined");
        }
    }

}

var tap = false;

function handleRegionTap(event) {
    /*
        Handles single and double tap in touch devices
    */
    if (debug) console.log("> handleRegionTap");

    var caller = this;

    if (!tap) { //if tap is not set, set up single tap
        tap = setTimeout(function () {
            tap = null
        }, 300);

        // call singlePressOnRegion(event) using 'this' as context
        singlePressOnRegion.call(this, event);
    } else {
        clearTimeout(tap);
        tap = null;

        // call doublePressOnRegion(event) using 'this' as context
        //doublePressOnRegion.call(this, event);
    }
    if (debug) console.log("< handleRegionTap");
}

function mouseDown(x, y) {
    if (debug > 1) console.log("> mouseDown");

    var prevRegion = null;
    var point = paper.view.viewToProject(new paper.Point(x, y));
    var imgCoord = viewer.viewport.windowToImageCoordinates(new OpenSeadragon.Point(x, y));
    handle = null;

    if (selectedTool == "select") {
        var hitResult = paper.project.hitTest(point, {
            //segments: true,
            stroke: true,
            fill: true,
            tolerance: 0.001
        });

        newRegionFlag = false;
        if (hitResult) {
            console.log(hitResult);
            var i;
            for (i = 0; i < ImageInfo[currentImage]["Regions"].length; i++) {
                var hitResultpath = hitResult.item;
                if (hitResult.item._parent instanceof paper.CompoundPath) {
                    hitResultpath = hitResult.item._parent;
                }
                if (ImageInfo[currentImage]["Regions"][i].path == hitResultpath) {
                    re = ImageInfo[currentImage]["Regions"][i];
                    break;
                }
            }

            // select path
            if (region && region != re) {
                if (region.path) {
                    region.path.selected = false;
                }
                prevRegion = region;
            }
            selectRegion(re);

            if (hitResult.type == 'handle-in') {
                handle = hitResult.segment.handleIn;
                handle.point = point;
            } else if (hitResult.type == 'handle-out') {
                handle = hitResult.segment.handleOut;
                handle.point = point;
            } else if (hitResult.type == 'segment') {
                handle = hitResult.segment.point;
                handle.point = point;
            }
        }
        if (hitResult == null && region) {
            //deselect paths
            if (region.path) {
                region.path.selected = false;
            }
            region = null;
        }
        if (hitResult == null) {
            deselectAllButtons();
        }
    } else if (selectedTool == "draw") {
        // Start a new region
        // if there was an older region selected, unselect it
        if (region && region.path) {
            region.path.selected = false;
        }
        // start a new region
        var path = new paper.Path({segments: [point]});
        path.strokeWidth = config.defaultStrokeWidth;
        region = newRegion({path: path, imgCoords: [imgCoord], zoom: getZoom()});
        // signal that a new region has been created for drawing
        newRegionFlag = true;
    }

    paper.view.draw();
}

function mouseDrag(x, y, dx, dy) {
    if (debug) console.log("> mouseDrag");

    // transform screen coordinate into world coordinate
    var point = paper.view.viewToProject(new paper.Point(x, y));
    var imgCoord = viewer.viewport.windowToImageCoordinates(new OpenSeadragon.Point(x, y));

    // transform screen delta into world delta
    var orig = paper.view.viewToProject(new paper.Point(0, 0));
    var dpoint = paper.view.viewToProject(new paper.Point(dx, dy));
    dpoint.x -= orig.x;
    dpoint.y -= orig.y;

    if (handle) {
        handle.x += point.x - handle.point.x;
        handle.y += point.y - handle.point.y;
        handle.point = point;
    } else if (selectedTool == "draw") {
        region.path.add(point);
        region.imgCoords.push(imgCoord)
    } else if (selectedTool == "select") {
        // event.stopHandlers = true;
        for (var i in ImageInfo[currentImage]["Regions"]) {
            var reg = ImageInfo[currentImage]["Regions"][i];
            if (reg.path) {
                if (!reg.name.toLowerCase().includes('colormap')) {

                    if (reg.path.selected) {
                        reg.path.position.x += dpoint.x;
                        reg.path.position.y += dpoint.y;
                    }
                }
            }
        }
    }

    paper.view.draw();
}


function isInsideCircularItem(x, y, path) {

    var segments = path.segments;
    //Test each segment for intersection
    var number_vertexes = segments.length;
    var contains = false;

    //Foreach segment, let's test intersection and flip the sign of the contains variable
    //Test from the test point going right
    for (var i = 0, j = number_vertexes - 1; i < number_vertexes; j = i++) {
        //If we're in the y range and we intersect, then that's what we want, flip the sign of contains
        if (((segments[i].point.y > y) != (segments[j].point.y > y)) && (x < (segments[j].point.x - segments[i].point.x) * (y - segments[i].point.y) / (segments[j].point.y - segments[i].point.y) + segments[i].point.x)) {
            contains = !contains;
        }
    }
    return contains;
}

function isInsideCircularPath(x, y, compoundPath) {

    var contains = false;
    if (compoundPath instanceof paper.CompoundPath) {
        for (var i = 0; i < compoundPath.children.length; i++) {
            if (isInsideCircularItem(x, y, compoundPath.children[i]) == true) {
                contains = true;
                break;
            }

        }
    } else if (compoundPath instanceof paper.Path) {
        contains = isInsideCircularItem(x, y, compoundPath);
    }
    return contains;
}

function solveSelfCrossings(region_index) {
    //CORRECCION DE REGIONES EN CASO DE SELF-CROSSING
    if (region.path.getCrossings().length > 0) //hay que resolver auto-intersecciones
    {
        //Al llamar a unite se gestionan las auto-intersecciones.
        var newPath_unite = ImageInfo[currentImage]["Regions"][region_index].path.unite();
        //Cuando la interseccion no está en un punto interior se genera un CompoundPath
        if (newPath_unite instanceof paper.CompoundPath) {
            //Sólo nos vamos a quedar con la region de área máxima
            var area, max_area = 0, indice_max_area = 0;
            for (var nr = 0; nr < newPath_unite._children.length; nr++) {
                area = Math.abs(newPath_unite.children[nr]._area);

                if (area > max_area) {
                    max_area = area;
                    indice_max_area = nr;
                }
            }
            ImageInfo[currentImage]["Regions"][region_index].path._segments =
                newPath_unite.children[indice_max_area]._segments;
            ImageInfo[currentImage]["Regions"][region_index].path._curves =
                newPath_unite.children[indice_max_area]._curves;
            for (var nr = 0; nr < newPath_unite._children.length; nr++) {
                newPath_unite.children[nr].remove();
            }
            newPath_unite.remove();

        } else {
            ImageInfo[currentImage]["Regions"][region_index].path.remove();
            ImageInfo[currentImage]["Regions"][region_index].path = newPath_unite;
        }

        ImageInfo[currentImage]["Regions"][region_index].imgCoords.splice(
            0, ImageInfo[currentImage]["Regions"][region_index].imgCoords.length);
        //ImageInfo[currentImage]["Regions"][region_index].zoom = getZoom();
        for (var np = 0; np < ImageInfo[currentImage]["Regions"][region_index].path.segments.length; np++) {
            ImageInfo[currentImage]["Regions"][region_index].imgCoords.push(convertPathToImgCoordinates(
                ImageInfo[currentImage]["Regions"][region_index].path._segments[np]._point))
        }
        region.path = ImageInfo[currentImage]["Regions"][region_index].path;

    }
}

function solveInterRegionCrossings() {

    var children = paper.project.activeLayer.children; //Todos los paths en la capa
    var intersection_performed = 0;
    var original;
    for (var i = 0; i < children.length; i++) {
        //Se recorren todos los paths buscando intersecciones entre las anotaciones existentes y la recién creada
        if (region.path != children[i] && intersection_performed == 0) {
            var intersections = region.path.getIntersections(children[i]);
            if (intersections.length == 2) {
                var region_index = getIndexFromPath(children[i]);
                if (ImageInfo[currentImage]["Regions"][region_index].name == region.name) {
                    original = ImageInfo[currentImage]["Regions"][region_index].path;
                    if (isInsideCircularPath(region.path.segments[0].point.x, region.path.segments[0].point.y, ImageInfo[currentImage]["Regions"][region_index].path)) {
                        var newPath = ImageInfo[currentImage]["Regions"][region_index].path.unite(region.path, {insert: true});
                    } else {
                        var newPath = ImageInfo[currentImage]["Regions"][region_index].path.subtract(region.path, {insert: true});
                    }
                    removeRegion(region);
                    ImageInfo[currentImage]["Regions"][region_index].path.remove();
                    ImageInfo[currentImage]["Regions"][region_index].path = newPath;
                    updateRegionImgCoords(region_index);
                    region = ImageInfo[currentImage]["Regions"][region_index];
                    paper.view.draw();
                    intersection_performed = 1;
                    break;

                }
            } else if (intersections.length > 2) {
                removeRegion(region);
                intersection_performed = -1;

            }
        }
    }
    return intersection_performed;


}

function solveItemHole(children, region_index) {

    var region_index_2 = getIndexFromPath(children);
    console.log("SOLVEITEMHOLE");
    console.log(ImageInfo[currentImage]["Regions"])
    console.log(region_index_2);

    var newPath = ImageInfo[currentImage]["Regions"][region_index_2].path.subtract(region.path, {insert: true});

    if (region_index_2 < ImageInfo[currentImage]["Regions"].length - 1) {
        removeRegion(region)
        ImageInfo[currentImage]["Regions"][region_index_2].path.remove();
        ImageInfo[currentImage]["Regions"][region_index_2].path = newPath;
        //Se actualizan las coordenadas reales
        updateRegionImgCoords(region_index_2);
    } else {
        removeRegion(ImageInfo[currentImage]["Regions"][region_index_2]);
        ImageInfo[currentImage]["Regions"][region_index].path.remove();
        ImageInfo[currentImage]["Regions"][region_index].path = newPath;
        //Se actualizan las coordenadas reales
        updateRegionImgCoords(region_index);
    }


    return region_index_2;

}

function solvePathHoles(region_index) {
    var flag_hole_solved = 1;
    var children = paper.project.activeLayer.children; //Todos los paths en la capa
    for (var k = 0; k < children.length && flag_hole_solved == 1; k++) {
        if (children[k] != region.path) {
            var region_index_2 = getIndexFromPath((children[k]));
            if (typeof region_index_2 !== 'undefined') {
                if (ImageInfo[currentImage]["Regions"][region_index_2].name == region.name) {
                    if (children[k] instanceof paper.CompoundPath) {
                        for (var j = 0; j < children[k].length && flag_hole_solved == 1; j++) {
                            if (isInsideCircularPath(
                                region.path.segments[0].point.x,
                                region.path.segments[0].point.y, children[k].children[j])) {
                                region_index = solveItemHole(children[k], region_index);
                                flag_hole_solved = 0;

                            }
                        }
                    } else {
                        if (isInsideCircularPath(region.path.segments[0].point.x, region.path.segments[0].point.y, children[k])) {
                            region_index = solveItemHole(children[k], region_index);
                            flag_hole_solved = 0
                        }
                    }
                }
            }
        }
    }
    region = ImageInfo[currentImage]["Regions"][region_index];
    return region_index;
}

function isTooSmall(region_index) {
    //CORRECCION PATHS-segments SIN PUNTOS o pocos segments
    var remove_region = false;

    for (var ss = 0; ss < ImageInfo[currentImage]["Regions"][region_index].path._segments.length; ss++) {
        if (!ImageInfo[currentImage]["Regions"][region_index].path._segments[ss].hasOwnProperty('_point')) {
            remove_region = true;
        }
    }
    if (ImageInfo[currentImage]["Regions"][region_index].path._segments.length <= 3) {
        remove_region = true;
    }
    if (Math.abs(ImageInfo[currentImage]["Regions"][region_index].path.area) < 0.03) {
        remove_region = true;
    }
    return remove_region;
}

function getIndexFromPath(path) {
    var region_index;
    for (var j = 0; j < ImageInfo[currentImage]["Regions"].length; j++) {
        if (ImageInfo[currentImage]["Regions"][j].path == path) {
            region_index = j;
            break
        }

    }
    return region_index;

}

function setRegionDepths() {
    var area = [];
    var indices = [];
    //Por area
    for (var j = 0; j < ImageInfo[currentImage]["Regions"].length; j++) {
        area[j] = ImageInfo[currentImage]["Regions"][j].path.area;
    }
    var indices = Array.from(Array(area.length).keys())
        .sort((a, b) => area[a] < area[b] ? -1 : (area[b] < area[a]) | 0);

    for (var j = 0; j < ImageInfo[currentImage]["Regions"].length; j++) {
        ImageInfo[currentImage]["Regions"][indices[j]].path.sendToBack();
    }

    sendRegionsToBack('ROI_Tumoral');
    sendRegionsToBack('ROI_NonTumoral');
    sendRegionsToBack('colormap');
}

function mouseUp() {

    if (debug) console.log("> mouseUp");

    var region_index;
    var remove_region;
    if (newRegionFlag == true && selectedTool == "draw") {
        region.path.closed = true;
        region_index = getIndexFromPath(region.path);
        remove_region = isTooSmall(region_index);

        if (remove_region == true) {
            ImageInfo[currentImage]["Regions"][region_index].path.remove();
            removeRegion(region);
        } else {
            if (region.name == 'ROI_Tumoral' || region.name == 'ROI_NonTumoral') {
                region.path.strokeColor = region.path.fillColor;
                region.path.strokeColor.alpha = 1;
                region.path.fillColor.alpha = 0;
                region.path.strokeWidth = 5;
                region.path.sendToBack();
            }
            if (region.path.getCrossings().length > 0) //hay que resolver auto-intersecciones
            {
                solveSelfCrossings(region_index);
                region = ImageInfo[currentImage]["Regions"][region_index];
            }
            var intersection_performed = solveInterRegionCrossings();
            if (intersection_performed == 0) {

                region_index = solvePathHoles(region_index);
                region = ImageInfo[currentImage]["Regions"][region_index];
            }
        }
    }
    console.log(selectedTool);
    if (selectedTool == "draw" && newRegionFlag == true) {
        if (remove_region == false) {
            selectRegion(region);
            appendRegionsToList();


        }
    }
    if (selectedTool == "select") {
        selectRegion(region);
        deselectAllButtons();
    }
    navEnabled = true;
    newRegionFlag = false;
    selectTool();
    paper.view.draw();
}

/***
 the following functions serve changing the annotation style
 ***/
var annotationColorLabel;

// add leading zeros
function pad(number, length) {
    var str = '' + number;
    while (str.length < length)
        str = '0' + str;
    return str;
}

/*** get current alpha & color values for colorPicker display
 ***/
function annotationStyle(label) {
    if (debug) console.log("> changing annotation style");

    annotationColorLabel = label;
    var alpha = label.alpha;
    $('#alphaSlider').val(alpha * 100);
    $('#alphaFill').val(parseInt(alpha * 100));

    var hexColor = '#' + pad((parseInt(label.color.red)).toString(16), 2) + pad((parseInt(label.color.green)).toString(16), 2) + pad((parseInt(label.color.blue)).toString(16), 2);
    if (debug) console.log(hexColor);

    $('#fillColorPicker').val(hexColor);

    if ($('#colorSelector').css('display') == 'none') {
        $('#colorSelector').css('display', 'block');
    } else {
        $('#colorSelector').css('display', 'none');
    }
}

/*** set picked color & alpha
 ***/
function setRegionColor() {
    var hexColor = $('#fillColorPicker').val();
    var red = parseInt(hexColor.substring(1, 3), 16);
    var green = parseInt(hexColor.substring(3, 5), 16);
    var blue = parseInt(hexColor.substring(5, 7), 16);
    var alpha = $('#alphaSlider').val() / 100;
    annotationColorLabel.alpha = alpha;
    // update region tag
    $(".region-tag#" + annotationColorLabel.uid + ">.region-color").css('background-color', 'rgba(' + red + ',' + green + ',' + blue + ',' + alpha + ')');
    $('#colorSelector').css('display', 'none');
}

/*** update all values on the fly
 ***/
function onFillColorPicker(value) {
    $('#fillColorPicker').val(value);
    var hexColor = $('#fillColorPicker').val();
    annotationColorLabel.color.red = parseInt(hexColor.substring(1, 3), 16);
    annotationColorLabel.color.green = parseInt(hexColor.substring(3, 5), 16);
    annotationColorLabel.color.blue = parseInt(hexColor.substring(5, 7), 16);
    annotationColorLabel.alpha = $('#alphaSlider').val() / 100;

    updateAnnotationStyle();

    paper.view.draw();
}

function onAlphaSlider(value) {
    $('#alphaFill').val(value);
    annotationColorLabel.alpha = $('#alphaSlider').val() / 100;

    updateAnnotationStyle();

    paper.view.draw();
}

function onAlphaInput(value) {
    $('#alphaSlider').val(value);
    annotationColorLabel.alpha = $('#alphaSlider').val() / 100;

    updateAnnotationStyle();

    paper.view.draw();
}

function updateAnnotationStyle() {
    for (var i = 0; i < ImageInfo[0].Regions.length; i++) {
        var reg = ImageInfo[0].Regions[i];
        if (reg.name == annotationColorLabel.label) {
            // change region color
            reg.path.fillColor = 'rgba(' + annotationColorLabel.color.red + ',' + annotationColorLabel.color.green +
                ',' + annotationColorLabel.color.blue + ',' + annotationColorLabel.alpha + ')';
        }
    }
}


function backToPreviousTool(prevTool) {
    setTimeout(function () {
        selectedTool = prevTool;
        selectTool()
    }, 500);
}

function backToSelect() {
    setTimeout(function () {
        selectedTool = "select";
        selectTool()
    }, 500);
}

/**
 * This function deletes the currently selected object.
 */
function cmdDeleteSelected() {
    var i;
    for (i in ImageInfo[currentImage]["Regions"]) {
        if (ImageInfo[currentImage]["Regions"][i] == region) {
            removeRegion(ImageInfo[currentImage]["Regions"][i]);
            paper.view.draw();
            break;
        }
    }
}

/**
 * This function shuffles the label of the currently selected object.
 */
function cmdShuffleSelected() {


    if (!$.isEmptyObject(region)) {
        var i;
        var indice = 0;

        for (i in ImageInfo[currentImage]["Regions"]) {
            if (ImageInfo[currentImage]["Regions"][i] == region) {
                indice = i;
                break
            }
        }
        if (getLabelIdByName(ImageInfo[currentImage]["Regions"][indice].name) > 0) {
            var select = $("<select>", {
                css: {
                    "width": "100%",
                    "margin-top": "1rem"
                }
            });

            for (var i = 0; i < labelDictionary.length; i++) {
                if (labelDictionary[i].uid > 2) {
                    select.append("<option>" + labelDictionary[i].label + "</option>");
                }

            }

            $.MessageBox({
                input: select,
                buttonDone: "OK",
                buttonFail: "Cancel",
                message: "Please, select a label from the list:",
                filterFail: function () {
                    navEnabled = true;
                }
            }).done(function (data) {
                if (!$.isEmptyObject(data)) {

                    var namei = ImageInfo[currentImage]["Regions"][indice].name;
                    var colori = getLabelColorByName(data);

                    ImageInfo[currentImage]["Regions"][indice].path.fillColor = new paper.Color(colori.red / 255.0, colori.green / 255.0, colori.blue / 255.0);
                    ImageInfo[currentImage]["Regions"][indice].path.fillColor.alpha = getLabelAlphaColorByName(data);

                    countLabel[namei] -= 1;
                    $('#count_' + labelIds[namei]).html(countLabel[namei]);
                    ImageInfo[currentImage]["Regions"][indice].name = data;
                    countLabel[data] += 1;
                    $('#count_' + labelIds[data]).html(countLabel[data]);
                    region.name = data;
                    selectRegion(region);
                    paper.view.draw(ImageInfo[currentImage]["Regions"][i]);


                } else {
                    navEnabled = true;
                }

            });
        }
    }
}

/**
 * This function segments a TUMORAL ROI automatically.
 */

function cmdAutoSelected() {
    var label_dialog = getNameFromCurrentStaining();

    if (currentStaining == 1) {
        if (!$.isEmptyObject(region)) {
            var i;
            var indice = 0;

            for (i in ImageInfo[currentImage]["Regions"]) {
                if (ImageInfo[currentImage]["Regions"][i] == region) {
                    indice = i;
                    break
                }
            }
            if (getLabelIdByName(ImageInfo[currentImage]["Regions"][indice].name) == 2) {

                var select = $("<select>", {
                    css: {
                        "width": "100%",
                        "margin-top": "1rem"
                    }
                });

                for (var i = 0; i < labelDictionary.length; i++) {
                    if (labelDictionary[i].uid > 2) {
                        select.append("<option>" + labelDictionary[i].label + "</option>");
                    }

                }
                $.MessageBox({
                    input: select,
                    buttonDone: "OK",
                    buttonFail: "Cancel",
                    message: "Please, assign a label from the list to the tumoral areas:",
                    filterFail: function () {
                        navEnabled = true;
                    }
                }).done(function (data) {
                    if (!$.isEmptyObject(data)) {

                        $.LoadingOverlay("show");
                        $.ajax({
                            type: "POST",
                            url: "/annotation/autoprocessing",
                            data: {
                                imgCoords: JSON.stringify(ImageInfo[currentImage]["Regions"][indice].imgCoords),
                                slide: slideName,

                            }
                        }).done(function (segmentationResult) {
                            toggleAllRegions();
                            toggleAllRegions();
                            addImportedRegions(segmentationResult, data, getZoom(), true);
                            $.LoadingOverlay("hide");
                            navEnabled = true;
                        });
                    } else {
                        navEnabled = true;
                    }
                });
            }
        }
    }
    //Ki67
    else if (currentStaining == 6 || currentStaining == 5 || currentStaining == 2 || currentStaining == 3 || currentStaining == 7 || currentStaining == 9) {
        $.MessageBox({
            buttonDone: "OK",
            buttonFail: "Cancel",
            message: "",
            input: {
                processingType: {
                    type: "select",
                    label: "Select the type of autoprocessing desired:",
                    title: "Select the type of autoprocessing desired",
                    options: {
                        "1": "Global " + label_dialog + " colormap",
                        "2": "Detailed ROI cell nuclei segmentation",
                    },
                    defaultValue: "1"
                },
            }
        }).done(function (data) {

            if (data.processingType == 2) { //Nuclei segmentation

                if (!$.isEmptyObject(region)) {
                    var i;
                    var indice = 0;

                    for (i in ImageInfo[currentImage]["Regions"]) {
                        if (ImageInfo[currentImage]["Regions"][i] == region) {
                            indice = i;
                            break
                        }
                    }
                    if (getLabelIdByName(ImageInfo[currentImage]["Regions"][indice].name) == 2) {

                        var select = $("<select>", {
                            css: {
                                "width": "100%",
                                "margin-top": "1rem"
                            }
                        });

                        for (var i = 0; i < labelDictionary.length; i++) {
                            if (labelDictionary[i].uid > 2) {
                                select.append("<option>" + labelDictionary[i].label + "</option>");
                            }

                        }
                        $.LoadingOverlay("show");

                        $.ajax({
                            type: "POST",
                            url: "/annotation/autoprocessing",
                            data: {
                                imgCoords: JSON.stringify(ImageInfo[currentImage]["Regions"][indice].imgCoords),
                                slide: slideName,
                                processingType: 'hotspot',
                            }

                        }).done(function (segmentationResult) {
                            segmentationResult = JSON.parse(segmentationResult)
                            region.score = segmentationResult.score;
                            toggleAllRegions();
                            toggleAllRegions();

                            addImportedRegions(segmentationResult.data, "", getZoom(), true);

                            $.LoadingOverlay("hide");
                        });

                    }
                }
            } else {  //Colormap
                $.LoadingOverlay("show");

                $.ajax({
                    type: "POST",
                    url: "/annotation/autoprocessing",
                    data: {
                        slide: slideName,
                        processingType: 'global',
                    }

                }).done(function (segmentationResult) {
                    toggleAllRegions();
                    toggleAllRegions();
                    addImportedRegions(segmentationResult, "", getZoom(), true);
                    $.LoadingOverlay("hide");
                });
            }

        });
    }
}

function cmdAddInfoSelected() {
    $.ajax({
        type: "POST",
        url: "/annotation/getWsiInfo",
        data: {
            slide: getSlideNameWithExt(),
        }
    }).done(function (out) {
        $.MessageBox({
            input: {
                text_cellularity: {
                    type: "text",
                    label: "Tumor cellularity (%):",
                    title: "Number from 0 to 100",
                    defaultValue: out.cellularity,
                    maxlength: 5,
                },
                text_comments: {
                    type: "textarea",
                    label: "Comments:",
                    title: "Input some other text",
                    resize: true,
                    defaultValue: out.comments,
                },

            },
            buttonDone: "OK",
            buttonFail: "Cancel",
            message: "Please, add some additional information:",
            filterFail: function () {
                navEnabled = true;
            }
        }).done(function (data) {
            if (!$.isEmptyObject(data)) {
                saveWsiInfo(data.text_cellularity, data.text_comments)
            } else {
                navEnabled = true;
            }
        });
    });
}

function clearToolSelection() {
    selectedTool = "navigate";
    selectTool();
    navEnabled = true;
    $('body').css('cursor', 'auto');
}

function toolSelection(event) {
    if (debug) console.log("> toolSelection");

    var prevTool = selectedTool;
    selectedTool = $(this).attr("id");
    selectTool();

    switch (selectedTool) {
        case "select":
        case "draw":
        case "navigate":
            navEnabled = true;
            handle = null;
            break;
        case "home":
            backToPreviousTool(prevTool);
            break;

    }
}

function selectTool() {
    if (debug) console.log("> selectTool");
    $("img.button").removeClass("selected");
    $("img.button#" + selectedTool).addClass("selected");
}


function saveWsiInfo(cellularity, comments) {
    $.ajax({
        type: "POST",
        url: "/annotation/saveWsiInfo",
        data: {
            slide: getSlideNameWithExt(),
            cellularity: cellularity,
            comments: comments,
        }
    }).done(function (out) {
    });

    if (debug) console.log("< saving additional info");
}

/***4
 Annotation storage
 */

function saveJson(json, annotationName, annotationType, validation) {
    if (debug) console.log("> writing json to file - debugging");

    $.ajax({
        type: "POST",
        url: "/annotation/saveJson",
        data: {
            json: JSON.stringify(json),
            slide: getSlideNameWithExt(),
            annotation_name: annotationName,
            annotation_type: annotationType,
            validation: validation,
        }
    }).done(function (annotationName) {
        currentAnnotation = annotationName;
        $('#currentAnnotationName').html(currentAnnotation);
    });

    if (debug) console.log("< writing json to file");

}


function getSlideNameWithExt() {
    var pos1 = slide.name.lastIndexOf("/");
    if (pos1 > 0) {
        return slide.name.slice(pos1 + 1);
    } else {
        return slide.name;
    }
}


/***5
 Initialisation
 */

function resizeAnnotationOverlay() {
    if (debug) console.log("> resizeAnnotationOverlay");

    var width = $("body").width();
    var height = $("body").height();
    $("canvas.overlay").width(width);
    $("canvas.overlay").height(height);
    paper.view.viewSize = [width, height];
}

function initAnnotationOverlay(data) {
    if (debug) console.log("> initAnnotationOverlay");

    // do not start loading a new annotation if a previous one is still being loaded
    if (annotationLoadingFlag == true) {
        return;
    }


    //console.log("new overlay size" + viewer.world.getItemAt(0).getContentSize());

    /*
       Activate the paper.js project corresponding to this slice. If it does not yet
       exist, create a new canvas and associate it to the new project. Hide the previous
       slice if it exists.
    */
    currentImage = 0;

    // change myOrigin (for loading and saving)
    myOrigin.slice = currentImage;

    // hide previous slice
    if (prevImage && paper.projects[ImageInfo[prevImage]["projectID"]]) {
        paper.projects[ImageInfo[prevImage]["projectID"]].activeLayer.visible = false;
        $(paper.projects[ImageInfo[prevImage]["projectID"]].view.element).hide();
    }

    // if this is the first time a slice is accessed, create its canvas, its project,
    // and load its regions from the database
    if (ImageInfo[currentImage]["projectID"] == undefined) {

        // create canvas
        var canvas = $("<canvas class='overlay' id='" + currentImage + "'>");
        $("body").append(canvas);

        // create project
        paper.setup(canvas[0]);
        ImageInfo[currentImage]["projectID"] = paper.project.index;

        if (debug) console.log('Set up new project, currentImage: ' + currentImage + ', ID: ' + ImageInfo[currentImage]["projectID"]);
    }

    // activate the current slice and make it visible
    paper.projects[ImageInfo[currentImage]["projectID"]].activate();
    paper.project.activeLayer.visible = true;
    $(paper.project.view.element).show();

    // resize the view to the correct size
    var width = $("body").width();
    var height = $("body").height();
    paper.view.viewSize = [width, height];
    paper.settings.handleSize = 5;
    paper.view.draw();

    /* RT: commenting this line out solves the image size issues */
    // set size of the current overlay to match the size of the current image
    //magicV = viewer.world.getItemAt(0).getContentSize().x / 100;
    //magicV = viewer.world.getItemAt(0).getContentSize().x;

    paper.project._needsUpdate = true;
    transform();
}

function transform() {
    //if( debug ) console.log("> transform");

    var z = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true));
    var sw = viewer.source.width;
    var bounds = viewer.viewport.getBounds(true);
    var x = magicV * bounds.x;
    var y = magicV * bounds.y;
    var w = magicV * bounds.width;
    var h = magicV * bounds.height;
    paper.view.setCenter(x + w / 2, y + h / 2);
    paper.view.zoom = (sw * z) / magicV;
    paper.view.draw();
}

function loadConfiguration() {
    var def = $.Deferred();
    // load general microdraw configuration
    $.getJSON(staticPath + "/configuration.json", function (data) {
        config = data;

        // set default values for new regions (general configuration)
        config.defaultStrokeColor = 'black';
        config.defaultStrokeWidth = 1;
        config.defaultFillAlpha = 0.5;

        // get list of dictionaries
        getDictionaryList();

        // load label dictionary
        loadDictionary(staticPath + "/dictionaries/" + slide.dictionary);

        drawingTools = ["select", "draw"];

        if (config.drawingEnabled == false) {
            // remove drawing tools from ui
            for (var i = 0; i < drawingTools.length; i++) {
                $("#" + drawingTools[i]).remove();
            }

        }
        for (var i = 0; i < config.removeTools.length; i++) {
            $("#" + config.removeTools[i]).remove();
        }

        def.resolve();
    });

    return def.promise();
}

function toggleDictPicker() {
    var dictListContent = $('#dicts_content');
    dictListContent.is(":visible") ? dictListContent.hide() : dictListContent.show();
}

function getDictionaryList() {
    $.ajax({
        url: "/annotation/getDictionaries",
        dataType: "json",
        success: function (localDicts) {
            var content = "";
            dictionaries = localDicts;

            for (var i in dictionaries) {
                content += '<p class="dictListEntry" onClick="dictListClick(' + i + ')">' + dictionaries[i] + '</p>';
            }
            $('#dicts_content').html(content);
        }
    });
}

function loadDictionary(path) {
    while (ImageInfo[currentImage].Regions.length > 0) {
        removeRegion(ImageInfo[currentImage].Regions[0])
    }
    $.ajax({
        url: path,
        dataType: "json",
        success: function (dictionary) {
            labelDictionary = dictionary;
            $('#currentDictName').html(slide.dictionary);
            appendLabelsToList();
        },
        error: function (data) {
            createNewDictionary(false);
        }
    });
}

function initAnnotationService() {

    if (debug) console.log("> initAnnotationService promise");

    var def = $.Deferred();

    // Enable click on toolbar buttons
    $("img.button").click(toolSelection);

    // set annotation loading flag to false
    annotationLoadingFlag = false;

    // Configure currently selected tool
    selectedTool = "navigate";
    selectTool();

    // set up the ImageInfo array and imageOrder array
    currentImage = 0;
    ImageInfo[currentImage] = {"Regions": [], "projectID": undefined};

    // load image viewer

    viewer = OpenSeadragon({
        id: "openseadragon1",
        prefixUrl: staticPath + "/lib/openseadragon/images/",
        showReferenceStrip: false,
        referenceStripSizeRatio: 0.2,
        showNavigator: true,
        sequenceMode: false,
        navigatorId: "myNavigator",
        homeButton: "home",
        preserveViewport: true,
        zoomPerClick: 1,
    });

    var mpp = 0;
    if (slide.mpp) {
        mpp = slide.mpp > 0 ? (1e6 / slide.mpp) : 0
    }

    viewer.scalebar({
        type: OpenSeadragon.ScalebarType.MICROSCOPE,
        minWidth: '150px',
        pixelsPerMeter: mpp,
        color: 'black',
        fontColor: 'black',
        backgroundColor: "rgba(255,255,255,0.5)",
        barThickness: 4,
        location: OpenSeadragon.ScalebarLocation.TOP_RIGHT,
        xOffset: 5,
        yOffset: 5
    });

    // open the currentImage
    viewer.open(slide.url);

    // add handlers: update slice name, animation, page change, mouse actions
    viewer.addHandler('open', function () {
        initAnnotationOverlay();
        $("title").text(slide.name);

        // todo: check if true:
        // To improve load times, ignore the lowest-resolution Deep Zoom
        // levels.  This is a hack: we can't configure the minLevel via
        // OpenSeadragon configuration options when the viewer is created
        // from DZI XML.
        viewer.source.minLevel = 8;


    });
    viewer.addHandler('animation', function (event) {
        transform();
    });
    viewer.addHandler("page", function (data) {
        console.log(data.page, params.tileSources[data.page]);
    });
    viewer.addViewerInputHook({
        hooks: [
            {tracker: 'viewer', handler: 'clickHandler', hookHandler: clickHandler},
            {tracker: 'viewer', handler: 'pressHandler', hookHandler: pressHandler},
            {tracker: 'viewer', handler: 'dragHandler', hookHandler: dragHandler},
            {tracker: 'viewer', handler: 'releaseHandler', hookHandler: releaseHandler},
            {tracker: 'viewer', handler: 'dragEndHandler', hookHandler: dragEndHandler}
        ]
    });
    // Show and hide menu
    if (config.hideToolbar) {
        var mouse_position;
        var animating = false;
        $(document).mousemove(function (e) {
            if (animating) {
                return;
            }
            mouse_position = e.clientX;

            if (mouse_position <= 100) {
                //SLIDE IN MENU
                animating = true;
                $('#menuBar').animate({
                    left: 0,
                    opacity: 1
                }, 200, function () {
                    animating = false;
                });
            } else if (mouse_position > 200) {
                animating = true;
                $('#menuBar').animate({
                    left: -100,
                    opacity: 0
                }, 500, function () {
                    animating = false;
                });
            }
        });
    }

    $(window).resize(function () {
        $("#regionList").height($(window).height() - $("#regionList").offset().top);
        resizeAnnotationOverlay();
    });

    return def.promise();
}

function help(show) {
    if (show) {
        $('#helpContent').show();
    } else {
        $('#helpContent').hide();
    }
}

function importInfo(show) {
    if (show) {
        $('#importContent').show();
    } else {
        $('#importContent').hide();
    }
}

function saveInfo(show) {
    if (show) {
        $('#saveContent').show();
    } else {
        $('#saveContent').hide();
    }
}

function backInfo(show) {
    if (show) {
        $('#backContent').show();
    } else {
        $('#backContent').hide();
    }
}

function selectInfo(show) {
    if (show) {
        $('#selectContent').show();
    } else {
        $('#selectContent').hide();
    }
}

function eraseInfo(show) {
    if (show) {
        $('#eraseContent').show();
    } else {
        $('#eraseContent').hide();
    }
}

function addinfoInfo(show) {
    if (show) {
        $('#addinfoContent').show();
    } else {
        $('#addinfoContent').hide();
    }
}

function drawInfo(show) {
    if (show) {
        $('#drawContent').show();
    } else {
        $('#drawContent').hide();
    }
}

function autoInfo(show) {
    if (show) {
        $('#autoContent').show();
    } else {
        $('#autoContent').hide();
    }
}

function shuffleInfo(show) {
    if (show) {
        $('#shuffleContent').show();
    } else {
        $('#shuffleContent').hide();
    }
}

function roiNTInfo(show) {
    if (show) {
        $('#roiNTContent').show();
    } else {
        $('#roiNTContent').hide();
    }
}

function roiTInfo(show) {
    if (show) {
        $('#roiTContent').show();
    } else {
        $('#roiTContent').hide();
    }
}

function getAnnotationTypeInputForSaveDialog() {
    var at;
    if (currentStaining == 1) {
        at = {

            type: "select",
            label: "Select the type of annotation:",
            title: "Select the type of annotation",
            options: {
                "1": "Detailed ROI segmentation for algorithm training",
                "2": "Marking of high tumor purity for NGS sequencing",
            },
            defaultValue: "1"
        }

    } else if(currentStaining == 6 || currentStaining == 5 || currentStaining == 2 || currentStaining == 3 || currentStaining == 7 || currentStaining == 9){
        at =
            {
                type: "select",
                label: "Select the type of annotation:",
                title: "Select the type of annotation",
                options: {
                    "3": "Colormap / Nuclei Segmentation",
                },
                defaultValue: "3"
            }


    }

    return at;
}

function save() {
    deselectAllButtons();
    var checked_format = "";

    $.MessageBox({
        buttonDone: "OK",
        buttonFail: "Cancel",
        message: "",
        input: {
            filename: {
                type: "text",
                label: "Filename (max 50 characters):",
                title: "Annotation filename:",
                defaultValue: slideName.substring(0, slideName.lastIndexOf(".")),
                maxlength: 50
            },
            annotationtype: getAnnotationTypeInputForSaveDialog(),
            validationcheckbox: {
                type: "checkbox",
                label: "Validation:",
                title: "The annotation of this image is complete."
            },
        }
    }).done(function (data) {
        if ($.trim(data)) {
            console.log(data);
            var checked_format = data.filename;
            if (data.filename.lastIndexOf(".") >= 0) {
                checked_format = data.filename.substring(0, data.lastIndexOf("."));
            }
            //Check annotation file
            $.ajax({
                type: "GET",
                url: "/annotation/checkAnnotationExistence?src=" + checked_format,
            }).done(function (out) {
                if (out == "False") {
                    saveJson(ImageInfo[currentImage]["Regions"], checked_format, data.annotationtype, data.validationcheckbox);
                    $.MessageBox("Saved!");
                    clearToolSelection();

                } else {
                    $.MessageBox({
                        message: "Overwrite the existing <b>" + checked_format + "</b> file of type <i>" + out + "</i>?",
                        buttonDone: "Yes",
                        buttonFail: "No",
                        filterFail: function () {
                            save();
                        }
                    }).done(function () {
                        console.log(data.annotationtype);
                        saveJson(ImageInfo[currentImage]["Regions"], checked_format, data.annotationtype, data.validationcheckbox);
                        $.MessageBox("Saved!");
                        clearToolSelection();
                    });
                }
            });
        }
    });


}


function closeAnnotationNav() {
    document.getElementById("annotationNav").style.width = "0%";
    clearToolSelection();
}


//GUI buttons

function deselectAllButtons() {
    buttonMode = false;
    document.getElementById("drawButton").style.filter = "brightness(100%)";
    document.getElementById("eraseButton").style.filter = "brightness(100%)";
    document.getElementById("selectButton").style.filter = "brightness(100%)";
    document.getElementById("roiNTButton").style.filter = "brightness(100%)";
    document.getElementById("roiTButton").style.filter = "brightness(100%)";
    document.getElementById("shuffleButton").style.filter = "brightness(100%)";
    document.getElementById("autoButton").style.filter = "brightness(100%)";
    $('body').css('cursor', 'auto');
    navEnabled = true;
    selectTool();
}

function guiButtonDown(html_id, cursor_name, tool_id) {
    console.log(buttonMode);
    console.log(prevGuiButton);
    if (buttonMode == true && tool_id == prevGuiButton) {
        buttonMode = false;
        prevGuiButton = '';
        clearToolSelection();
        deselectAllButtons();
    } else {
        prevGuiButton = tool_id;
        if (tool_id == 'erase') {
            buttonMode = false;
            cmdDeleteSelected();
            appendRegionsToList();
            deselectAllButtons();
        } else if (tool_id == 'shuffle') {
            buttonMode = false;
            cmdShuffleSelected();
            appendRegionsToList();
            deselectAllButtons();

        } else if (tool_id == 'auto') {
            buttonMode = false;
            cmdAutoSelected();
            appendRegionsToList();
            deselectAllButtons();
            clearToolSelection();
        } else if (tool_id == 'addinfo') {
            buttonMode = false;
            cmdAddInfoSelected();
            appendRegionsToList();
            deselectAllButtons();
            clearToolSelection();
        } else {
            deselectAllButtons();
            buttonMode = true;
            document.getElementById(html_id).style.filter = "brightness(150%)";
            if (cursor_name.length > 0) {
                $('body').css('cursor', 'url(/static/cursors/' + cursor_name + '),auto');
            }
            selectToolOnKeyPress(tool_id);
        }

    }
}


// key listener
$(document).keydown(function (e) {

    deselectAllButtons();
    if (e.keyCode) code = e.keyCode;
    else if (e.which) code = e.which;
    var character = String.fromCharCode(code);

    if (e.keyCode == 9) {
        // tab
        e.preventDefault();
        selectNextRegionLabel();
    } else if (e.keyCode == 17) {
        // ctrl
        $('body').css('cursor', 'url(/static/cursors/editinibica_cursor.svg),auto');
        selectToolOnKeyPress("draw");
    } else if (e.keyCode == 18 || e.keyCode == 225) {
        // alt || alt gr
        $('body').css('cursor', 'url(/static/cursors/selectinibica_cursor.svg),auto');
        selectToolOnKeyPress("select");
    } else if (character == 'N') {
        $('body').css('cursor', 'url(/static/cursors/rointinibica_cursor.svg),auto');
        selectToolOnKeyPress("roib");
    } else if (character == 'T') {
        $('body').css('cursor', 'url(/static/cursors/roitinibica_cursor.svg),auto');
        selectToolOnKeyPress("roim");
    } else if (e.keyCode == 27) {
        // esc
        var dictListContent = $('#dicts_content');
        if (dictListContent.is(":visible")) {
            dictListContent.hide()
        } else {
            deselectRegion(region);
        }
    } else if (e.keyCode == 46) {
        //supr
        cmdDeleteSelected();
        appendRegionsToList();
    }
});

function selectToolOnKeyPress(id) {

    tmpTool = selectedTool;
    selectedTool = id;
    navEnabled = false;

    selectTool();
}

function createNewDictionary(isCancelable) {
    // get name for new dictionary
    var name = "";
    while (name.length == 0) {
        name = prompt("Enter new name for new dictionary", "dictionary");
        if (name === null) {
            // user hits "cancel" in prompt
            if (isCancelable) {
                name = null;
                break;
            } else {
                name = "";
                alert("No dictionary found. Please enter a valid name to create one.");
            }

        }
    }
    if (name) {
        if (name.indexOf(".json") === -1) {
            name = name + ".json"
        }
        // request creation of new dictionary and path to it
        $.ajax({
            type: "GET",
            url: "/annotation/createDictionary?name=" + name + "&slide=" + slide.name,
        }).done(function (response) {
            if (response === "error") {
                alert("Couldn't create dictionary since there is already a dictionary with that name!");
            } else {
                var json = JSON.parse(response);
                var path = json["path"];
                var name = json["name"];
                slide.dictionary = name;
                loadDictionary(path);
                getDictionaryList();
            }
        });
    }
}

$(document).keyup(function (e) {
    if (buttonMode == false) {

        if (e.keyCode) code = e.keyCode;
        else if (e.which) code = e.which;
        var character = String.fromCharCode(code);

        if (e.keyCode == 16 || e.keyCode == 17 || e.keyCode == 18 || e.keyCode == 225 || character == 'N' || character == 'T') {
            $('body').css('cursor', 'auto');
            // shift || ctrl || alt || alt gr
            if (e.keyCode == 17 && newRegionFlag == true) {
                mouseUp();
            }
            navEnabled = true;
            selectTool();
        }
    }

});

function init(file_name, annotation_path, url, mpp, dictionary) {

    currentImage = file_name;
    slideName = file_name;
    currentAnnotation = annotation_path;
    $('#currentWSIName').html(currentImage);
    file_name = '/annotation/' + file_name;
    slide = {"name": file_name, "url": url, "mpp": mpp, "dictionary": dictionary};
    staticPath = "/static";

    $.LoadingOverlay("show");
    loadConfiguration();
    initAnnotationService();


    if (currentAnnotation != "None") {
        var count = 0;
        var interval = setInterval(function () {
            if (count >= 100) {
                clearInterval(interval);
                importAnnotation(currentAnnotation);
                return;
            }
            count += 10;
        }, 100);

    } else {
        $('#currentAnnotationName').html(currentAnnotation);
        $.LoadingOverlay("hide");
    }

    $.ajax({
        type: "POST",
        url: "/annotation/getWsiInfo",
        data: {
            slide: getSlideNameWithExt(),
        }
    }).done(function (out) {
        currentStaining = out.staining_id;
    });


}
