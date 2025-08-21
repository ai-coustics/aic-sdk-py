# Low-level C bindings (`aic._bindings`)

The high-level `aic.Model` API is recommended for most applications. The low-level bindings mirror the C API and are useful for advanced integrations, benchmarks, or when you need fine-grained control.

## Core functions

::: aic._bindings.model_create
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.model_destroy
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.model_initialize
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.model_reset
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

## Processing

::: aic._bindings.process_planar
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.process_interleaved
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

## Parameters

::: aic._bindings.set_parameter
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.get_parameter
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

## Information helpers

::: aic._bindings.get_processing_latency
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.get_output_delay
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.get_optimal_sample_rate
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.get_optimal_num_frames
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.get_library_version
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

## Enums

::: aic._bindings.AICErrorCode
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.AICModelType
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false

::: aic._bindings.AICParameter
    options:
      show_root_heading: true
      heading_level: 3
      show_object_full_path: false