// ============================================================
// STEP D (optional): Sentinel-1 time series at the labelled points
// Exports VV and VH for every acquisition date (Jun 2024 - May 2025) so the
// classifier can use the phenological signal (e.g. paddy flooding) instead of
// a single annual median. Bands are selected and renamed BY NAME and per date,
// so VV and VH can never be swapped.
// ============================================================
var pts = ee.FeatureCollection('projects/YOUR_PROJECT/assets/cauvery_interpreted');
var studyArea = ee.Geometry.Rectangle([79.05, 10.85, 79.45, 11.15]);

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(studyArea).filterDate('2024-06-01', '2025-06-01')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);

// one mosaic per acquisition date (adjacent frames of the same pass are merged)
var dates = s1.aggregate_array('system:time_start')
  .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }).distinct().sort();
print('Acquisition dates:', dates);

var stacked = ee.Image(dates.iterate(function (d, acc) {
  var day = ee.Date.parse('YYYY-MM-dd', d);
  var tag = day.format('YYYYMMdd');
  var im = s1.filterDate(day, day.advance(1, 'day')).mosaic()
    .focalMedian({radius: 30, kernelType: 'circle', units: 'meters'})
    .select(['VV', 'VH'])
    .rename([ee.String('VV_').cat(tag), ee.String('VH_').cat(tag)]);
  return ee.Image(acc).addBands(im);
}, ee.Image().select([])));
print('Bands:', stacked.bandNames());

var samples = stacked.sampleRegions({
  collection: pts,
  properties: ['PLOTID', 'point_id', 'class', 'class_name'],
  scale: 10,
  geometries: true
});
print('Samples:', samples.size(), samples.first());

Export.table.toDrive({
  collection: samples, description: 'cauvery_S1_timeseries', fileFormat: 'CSV'
});
