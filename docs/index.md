# Dataframes File Interchange

For data pipelines, it's often advantageous to be able to "round trip" a dataframe to disc: the results of what is proccessed at one stage being required by independent stages afterwards.

Pandas can store and manipulate various structures on top of a basic table including indexes and dtype specifications. These indexes can be generated from a small number of parameters that cannot naturally be encoded in a column, e.g. a `pd.RangeIndex`. When saving to a CSV file, these indexes are typically enumerated. Therefore, when reading the dataframe back, it is not actually the same as the original. Although this is handled better when writing to Parquet format, there are still a few issues such as all entries in a column, and so an index, being required to have the same dtype. Similarly, it's often desirable to store some custom structured metadata along with the dataframe, e.g. the author or columnwise metadata such as what unit each column is expressed in.

`df_file_interchange` is a wrapper around Pandas that tries to address these requirements. It stores additional metadata in an accompanying YAML file that (mostly) ensures the indexes are recreated properly, column dtypes are specified properly, and that the dataframe read from disc is actually the same as what was written. It also manages the storage of additional (structured) custom metadata.

In the future, the intention is also to opportunistically serialise columns when required since there are some dtypes that are not supported by Parquet.





