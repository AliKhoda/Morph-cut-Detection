var project = app.project;

function findItem(name) {
    var item;
    for (var i = 0; i < project.rootItem.children.length; i++){
        var j = project.rootItem.children[i];
        if (!j.isSequence() && j.name == name){
            item = j;
            break;
        }
    }
    return item;
}
function insertClip(clip, inP, outP) {
    clip.setInPoint(inP,0);
    clip.setOutPoint(outP,0);
    activeSeq.videoTracks[0].insertClip(clip,0);
}
function createSequence(clip, inP, outP){
    clip.setInPoint(inP,0);
    clip.setOutPoint(outP,0);
    project.createNewSequenceFromClips(name+'_sequence', [clip]); // Create new sequence using projectitems
}
function addTransitions(){
    var file = new File("Morph Cut\\extendscripts\\hello.au3");
    file.execute();
    $.sleep(1000);
}

var reg = new RegExp('\\n','g');

var dataFile = File((new File($.fileName)).parent + "\\videos\\videos.txt");
dataFile.open("r");
var line = dataFile.readln();
while (line != ''){
    var videoFile = File((new File($.fileName)).parent + ("\\videos\\"+ line +".txt"));
    videoFile.open("r");
    var name = videoFile.readln();
    var path = (new File($.fileName)).parent + ("\\videos\\"+name);
    var in1 = parseFloat(videoFile.readln().replace(reg,''));
    var out1 = parseFloat(videoFile.readln().replace(reg,''));
    var in2 = parseFloat(videoFile.readln().replace(reg,''));
    var out2 = parseFloat(videoFile.readln().replace(reg,''));
    var in3 = parseFloat(videoFile.readln().replace(reg,''));
    var out3 = parseFloat(videoFile.readln().replace(reg,''));

    project.importFiles([path]); // Import file/files to media bucket
    var clip = findItem(name);
    if (typeof clip === 'undefined'){
        $.writeln('undefined');
    }
    else
    {
        createSequence(clip,in3,out3);
        $.sleep(1000);
        var activeSeq = app.project.activeSequence;
        insertClip(clip,in2,out2);
        insertClip(clip,in1,out1);
        addTransitions();
        activeSeq.setInPoint(0.5);
        activeSeq.setOutPoint(out3-in1-(in3-out2)-(in2-out1)-0.5);
        activeSeq.exportAsMediaDirect('Morph Cut\\morphcut_banners\\'+name, 'Morph Cut\\premiere-preset.epr', app.encoder.ENCODE_IN_TO_OUT);
        $.sleep(120000);
        activeSeq.exportAsMediaDirect('Morph Cut\\morphcuts\\'+name, 'Morph Cut\\premiere-preset.epr', app.encoder.ENCODE_IN_TO_OUT);
        project.deleteSequence(activeSeq);
    }

    line = dataFile.readln();
}
