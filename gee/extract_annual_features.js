// ============================================================
// STEP C: extract features at the interpreted points
// Input: cauvery_interpreted.csv uploaded as a table asset
//   (PLOTID, point_id, longitude, latitude, class 1 Rice / 2 Vegetation / 3 Built-up,
//    class_name, interpreter, labelled_at). Other/Unclear rows already removed.
// ============================================================
var interpreted = ee.FeatureCollection('projects/YOUR_PROJECT/assets/cauvery_interpreted');
var studyArea = ee.Geometry.Rectangle([79.05, 10.85, 79.45, 11.15]);

// The asset was uploaded with longitude/latitude as X/Y columns, so each
// feature already carries its point geometry - no need to rebuild it.
print('Interpreted points:', interpreted.size(), interpreted.first());

// Sentinel-2 NDVI
function maskS2(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start']);
}
var ndvi = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyArea).filterDate('2024-06-01', '2025-06-01')
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20)).map(maskS2)
  .median().normalizedDifference(['B8', 'B4']).rename('NDVI');

// Sentinel-1 (bands selected BY NAME; 30 m focal median per scene; temporal median)
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(studyArea).filterDate('2024-06-01', '2025-06-01')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH'])
  .map(function (img) {
    return img.focalMedian({radius: 30, kernelType: 'circle', units: 'meters'})
              .copyProperties(img, ['system:time_start']);
  });
print('Sentinel-1 scenes:', s1.size());
var m = s1.median();
var vv = m.select('VV'), vh = m.select('VH');
var feats = ndvi.addBands(vh.rename('VH')).addBands(vv.rename('VV'))
  .addBands(vv.divide(vh).rename('VV_VH_ratio'))          // same definition as the original corpus
  .addBands(vv.subtract(vh).rename('VV_minus_VH_dB'));    // conventional ratio in dB, for reference

var samples = feats.sampleRegions({
  collection: interpreted,
  properties: ['PLOTID', 'point_id', 'class', 'class_name', 'interpreter', 'labelled_at', 'longitude', 'latitude'],
  scale: 10,
  geometries: true
});
print('Samples with features:', samples.size());
print('Class distribution:', samples.aggregate_histogram('class'));

Export.table.toDrive({
  collection: samples, description: 'cauvery_features_labels_geometry', fileFormat: 'CSV',
  selectors: ['PLOTID', 'point_id', 'class', 'class_name', 'NDVI', 'VH', 'VV', 'VV_VH_ratio',
              'VV_minus_VH_dB', 'longitude', 'latitude', 'interpreter', 'labelled_at', '.geo']
});
