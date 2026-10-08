// ============================================================
// point labelling tool (GEE Code Editor)
// Shows each sample point on high-resolution satellite imagery with
// Sentinel-2 seasonal composites; click a class button to label and
// jump to the next point. Export your labels with the Export button.
// ============================================================

// ---------- SETTINGS (edit these 3 lines) ----------
var ASSET = 'projects/YOUR_PROJECT/assets/CEO_plots_500';
var INTERPRETER = 'RK';   // your initials
var START = 1;            // plot number to start from (change when resuming)
// ----------------------------------------------------

var studyArea = ee.Geometry.Rectangle([79.05, 10.85, 79.45, 11.15]);
Map.setOptions('SATELLITE');   // high-resolution Google satellite basemap

// Sentinel-2 true-colour composites for the seasonal check
function s2(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(studyArea).filterDate(start, end)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(function (img) {
      var qa = img.select('QA60');
      return img.updateMask(qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0)));
    })
    .median().clip(studyArea);
}
var vis = {bands: ['B4', 'B3', 'B2'], min: 0, max: 2500};
Map.addLayer(s2('2024-10-01', '2025-01-01'), vis, 'Sentinel-2 Oct-Dec 2024 (paddy season)', false);
Map.addLayer(s2('2025-03-01', '2025-06-01'), vis, 'Sentinel-2 Mar-May 2025 (dry season)', false);
var MARKER_INDEX = Map.layers().length();          // marker layer goes on top
Map.layers().add(ui.Map.Layer(ee.Image(), {}, 'current point'));

// ---------- state ----------
var plots = [];          // [{id, pid, lon, lat}]
var labels = {};         // PLOTID -> {class_name, confidence, labelled_at}
var idx = 0;
var confidence = '3';

// ---------- UI ----------
var title = ui.Label('Loading points…', {fontWeight: 'bold', fontSize: '16px'});
var status = ui.Label('');
var confSel = ui.Select({items: ['3 (certain)', '2 (fairly sure)', '1 (unsure)'],
  value: '3 (certain)', onChange: function (v) { confidence = v.charAt(0); }});

function classButton(name, color) {
  return ui.Button({label: name, style: {color: color, width: '130px'},
    onClick: function () { record(name); }});
}
var panel = ui.Panel({style: {width: '320px', position: 'top-left'}});
panel.add(title);
panel.add(ui.Label('Label what is UNDER the red square (10 m pixel).'));
panel.add(ui.Label('Confidence:')); panel.add(confSel);
panel.add(classButton('Rice', 'blue'));
panel.add(classButton('Vegetation', 'green'));
panel.add(classButton('Built-up', 'red'));
panel.add(classButton('Other', 'black'));
panel.add(classButton('Unclear', 'gray'));
panel.add(ui.Panel([
  ui.Button('< Back', function () { if (idx > 0) { idx--; show(); } }),
  ui.Button('Skip >', function () { if (idx < plots.length - 1) { idx++; show(); } })
], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Button({label: 'EXPORT labels so far', style: {fontWeight: 'bold'}, onClick: exportLabels}));
panel.add(status);
panel.add(ui.Label('Tip: toggle the Sentinel-2 layers in the Layers menu (top right) to check the season.',
  {fontSize: '11px'}));
Map.add(panel);

function show() {
  var p = plots[idx];
  var done = Object.keys(labels).length;
  var prev = labels[p.id] ? ' — already: ' + labels[p.id].class_name : '';
  title.setValue('Plot ' + p.id + ' / ' + plots[plots.length - 1].id + '  (' + p.pid + ')' + prev);
  status.setValue('Labelled: ' + done + '. Export regularly!');
  var pt = ee.Geometry.Point([p.lon, p.lat]);
  var square = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(pt.buffer(5).bounds())]), 1, 2);
  Map.layers().set(MARKER_INDEX, ui.Map.Layer(square, {palette: ['ff0000']}, 'current point'));
  Map.setCenter(p.lon, p.lat, 18);
}

function record(name) {
  var p = plots[idx];
  labels[p.id] = {class_name: name, confidence: confidence, labelled_at: new Date().toISOString()};
  if (idx < plots.length - 1) { idx++; show(); }
  else { status.setValue('Last point done — click EXPORT.'); }
}

function exportLabels() {
  var feats = [];
  plots.forEach(function (p) {
    var l = labels[p.id];
    if (l) {
      feats.push(ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {
        PLOTID: p.id, point_id: p.pid, longitude: p.lon, latitude: p.lat,
        class_name: l.class_name, confidence: l.confidence, interpreter: INTERPRETER,
        labelled_at: l.labelled_at, imagery: 'Google satellite basemap (GEE) + Sentinel-2 Oct-Dec 2024 / Mar-May 2025'
      }));
    }
  });
  if (feats.length === 0) { status.setValue('Nothing labelled yet.'); return; }
  var ids = feats.length ? Object.keys(labels).map(Number) : [];
  var name = 'cauvery_labels_' + INTERPRETER + '_' + Math.min.apply(null, ids) + '_' + Math.max.apply(null, ids);
  Export.table.toDrive({
    collection: ee.FeatureCollection(feats), description: name, fileFormat: 'CSV',
    selectors: ['PLOTID', 'point_id', 'longitude', 'latitude', 'class_name', 'confidence',
                'interpreter', 'labelled_at', 'imagery']
  });
  status.setValue('Export task "' + name + '" created: open the Tasks tab (top right) and click RUN.');
}

// ---------- load points ----------
// Coordinates are read from the LON / LAT columns of the table itself,
// so the tool works whether or not the asset was uploaded with point geometry.
ee.FeatureCollection(ASSET).evaluate(function (fc, err) {
  if (err) { title.setValue('Could not load asset: ' + err); return; }
  var bad = 0;
  fc.features.forEach(function (f) {
    var pr = f.properties;
    var lon = Number(pr.LON !== undefined ? pr.LON : pr.longitude);
    var lat = Number(pr.LAT !== undefined ? pr.LAT : pr.latitude);
    if ((isNaN(lon) || isNaN(lat)) && f.geometry && f.geometry.coordinates) {
      lon = Number(f.geometry.coordinates[0]); lat = Number(f.geometry.coordinates[1]);
    }
    if (isNaN(lon) || isNaN(lat)) { bad++; return; }
    plots.push({id: Number(pr.PLOTID), pid: pr.point_id, lon: lon, lat: lat});
  });
  plots.sort(function (a, b) { return a.id - b.id; });
  print('Points loaded: ' + plots.length + (bad ? ' (skipped ' + bad + ' without coordinates)' : ''));
  if (plots.length === 0) {
    title.setValue('No points with coordinates found. Columns in the asset: ' +
                   (fc.features.length ? Object.keys(fc.features[0].properties).join(', ') : 'none'));
    return;
  }
  idx = Math.max(0, Math.min(plots.length - 1, START - 1));
  show();
});
