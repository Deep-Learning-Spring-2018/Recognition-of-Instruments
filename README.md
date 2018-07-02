# Recognition-of-Instruments

This project use CNN to classify different instruments.

## How to prepated raw data?

1. Download from [download site](http://theremin.music.uiowa.edu/MIS-Pitches-2012/MISViolin2012.html)

2. enter the root directory of the project, unzip the download file and place them in `./resource` like the following structure.
```
ltoSax.NoVib.ff.stereo.zip    EbClarinet                     Oboe                         Viola.arco.ff.sulC.stereo.zip
AltoSax.vib.ff.stereo.zip      EbClarinet.ff.stereo.zip       Oboe.ff.stereo.zip           Viola.arco.ff.sulD.stereo.zip
BassFlute                      flute                          SopSax                       Viola.arco.ff.sulG.stereo.zip
BassFlute.ff.stereo.zip        flute.nonvib.ff.zip            SopSax.nonvib.ff.stereo.zip  violin
BassTrombone                   flute.nonvib.mf.zip            SopSax.vib.ff.stereo.zip     Violin
BassTrombone.ff.stereo.zip     flute.nonvib.pp.zip            TenorTrombone                Violin.arco.ff.sulA.stereo.zip
BbClarinet                     flute.nonvib.zip               TenorTrombone.ff.stereo.zip  Violin.arco.ff.sulD.stereo.zip
BbClarinet.ff.stereo.zip       flute.vib.ff.zip               Trumpet                      Violin.arco.ff.sulE.stereo.zip
Cello                          flute.vib.mf.zip               Trumpet.novib.ff.stereo.zip  Violin.arco.ff.sulG.stereo.zip
Cello.arco.ff.sulA.stereo.zip  flute.vib.pp.zip               Trumpet.vib.ff.stereo.zip
Cello.arco.ff.sulC.stereo.zip  flute.vib.zip                  unzip.ipynb
Cello.arco.ff.sulD.stereo.zip  Horn                           Viola
```

which means that you put all the audio belongs to an instrument to a directory named as this instrument.

In fact that the `unzip.ipynb` will help you will this process.

## How to prepocess data and train the model?

Just run `./run.sh` in the root directory, everything is automatically done.

## Dependency

1. `python3 >= 3.6.3`
2. `tensorflow > 1.6.0`
3. `tensorlayer`
4. `pysoundfile`
5. `numpy`
6. `scipy`
7. `matplotlib`
