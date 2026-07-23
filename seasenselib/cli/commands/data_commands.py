"""
Data processing commands (convert, show, subset, calc).
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from ...core.exceptions import ValidationError
from .base import BaseCommand, CommandResult

logger = logging.getLogger(__name__)


def _parse_reader_arg_value(value: str):
    """Parse simple CLI reader option values into useful Python types."""
    value = value.strip()
    lower = value.lower()

    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None

    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _build_reader_kwargs(args):
    """Build reader kwargs from CLI arguments."""
    reader_kwargs = {
        "sanitize_input": True,
        "fix_missing_coords": True,
    }

    # Parse generic reader args
    for item in getattr(args, "reader_args", None) or []:
        if "=" not in item:
            raise ValidationError(
                f"Invalid reader argument: {item}. Use NAME=VALUE."
            )
        name, value = item.split("=", 1)
        name = name.strip().replace("-", "_")
        if not name or not name.isidentifier():
            raise ValidationError(
                f"Invalid reader argument name: {name!r}."
            )
        reader_kwargs[name] = _parse_reader_arg_value(value)

    # Apply explicit flags last so they always win on conflicts
    if getattr(args, "no_sanitize", False):
        reader_kwargs["sanitize_input"] = False
    if getattr(args, "no_fix_coords", False):
        reader_kwargs["fix_missing_coords"] = False

    return reader_kwargs


def _build_stage_kwargs(args):
    """Build processing stage kwargs from CLI arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns
    -------
    dict
        Dictionary with processing step control parameters
        
    Raises
    ------
    ValueError
        If conflicting flags are specified
    """
    # Validate: --raw-only cannot be combined with other stage controls
    if getattr(args, 'raw_only', False):
        if (getattr(args, 'pipeline_apply_stages', None) or getattr(args, 'pipeline_skip_stages', None)
                or getattr(args, 'pipeline_profile', None)
                or getattr(args, 'pipeline_file', None)
                or getattr(args, 'pipeline_apply_handlers', None)
                or getattr(args, 'pipeline_skip_handlers', None)):
            raise ValueError(
                "--raw-only cannot be combined with pipeline stage/handler controls or pipeline file/profile"
            )

    # Validate: pipeline profile is exclusive with apply/skip
    if getattr(args, 'pipeline_profile', None):
        if getattr(args, 'pipeline_file', None):
            raise ValueError("--pipeline-profile cannot be combined with --pipeline-file")
        if getattr(args, 'pipeline_apply_stages', None) or getattr(args, 'pipeline_skip_stages', None):
            raise ValueError(
                "--pipeline-profile cannot be combined with --pipeline-apply-stages or --pipeline-skip-stages"
            )
    if getattr(args, 'pipeline_file', None):
        if getattr(args, 'pipeline_apply_stages', None) or getattr(args, 'pipeline_skip_stages', None):
            raise ValueError(
                "--pipeline-file cannot be combined with --pipeline-apply-stages or --pipeline-skip-stages"
            )
    
    stage_kwargs = {}
    
    # Check for --raw-only flag
    if getattr(args, 'raw_only', False):
        stage_kwargs['use_steps'] = False
    
    # Check for --pipeline-profile
    elif getattr(args, 'pipeline_profile', None):
        try:
            from ...pipeline.config import PipelineConfig
            config = PipelineConfig.from_resource(args.pipeline_profile)
            stage_kwargs['pipeline_config'] = config
        except Exception as e:
            raise ValueError(f"Unknown pipeline profile: {args.pipeline_profile}") from e
    # Check for --pipeline-file
    elif getattr(args, 'pipeline_file', None):
        try:
            from ...pipeline.config import PipelineConfig
            config = PipelineConfig.from_file(args.pipeline_file)
            stage_kwargs['pipeline_config'] = config
        except Exception as e:
            raise ValueError(f"Failed to load pipeline file: {args.pipeline_file}") from e

    # Check for --pipeline-apply-stages argument (explicit stage list)
    elif getattr(args, 'pipeline_apply_stages', None):
        stage_names = [s.strip() for s in args.pipeline_apply_stages.split(',')]
        # Build a pipeline config with only the specified steps
        try:
            from ...pipeline.config import PipelineConfig
            config = PipelineConfig()
            for stage_name in stage_names:
                config.add_stage(stage_name)
            stage_kwargs['pipeline_config'] = config
        except Exception as e:
            raise ValueError("Failed to build pipeline config from --pipeline-apply-stages") from e
    
    # Check for --pipeline-skip-stages argument
    elif getattr(args, 'pipeline_skip_stages', None):
        skip = [s.strip() for s in args.pipeline_skip_stages.split(',')]
        try:
            from ...pipeline.config import PipelineConfig
            config = PipelineConfig.from_resource("default")
            config.pipeline = [
                stage for stage in config.pipeline
                if stage.name not in skip
            ]
            stage_kwargs['pipeline_config'] = config
        except Exception:
            # If registry fails, just pass the skip info and let read() handle it
            logger.debug("Failed to resolve pipeline stages for --pipeline-skip-stages", exc_info=True)

    # Apply handler filters if provided
    apply_handlers = getattr(args, 'pipeline_apply_handlers', None)
    skip_handlers = getattr(args, 'pipeline_skip_handlers', None)
    if apply_handlers or skip_handlers:
        from ...pipeline.utils import parse_handler_selectors, apply_handler_filters
        apply_map = parse_handler_selectors(apply_handlers) if apply_handlers else {}
        skip_map = parse_handler_selectors(skip_handlers) if skip_handlers else {}
        if 'pipeline_config' not in stage_kwargs:
            from ...pipeline.config import PipelineConfig
            stage_kwargs['pipeline_config'] = PipelineConfig.from_resource("default")
        stage_kwargs['pipeline_config'] = apply_handler_filters(
            stage_kwargs['pipeline_config'], apply_map, skip_map
        )
    
    return stage_kwargs


def _parse_user_metadata(args):
    """Parse user metadata from CLI args."""
    from ...pipeline.utils import normalize_user_metadata, merge_user_metadata

    metadata = None

    metadata_file = getattr(args, 'metadata_file', None)
    if metadata_file:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metadata = normalize_user_metadata(data)
        except Exception as e:
            raise ValueError(f"Failed to load metadata file: {metadata_file}") from e

    metadata_inline = getattr(args, 'metadata', None)
    if metadata_inline:
        try:
            data = json.loads(metadata_inline)
        except Exception as e:
            raise ValueError("Invalid JSON provided via --metadata") from e
        metadata = merge_user_metadata(metadata, data)

    return metadata


def _build_writer_kwargs(args):
    """Build writer kwargs from CLI arguments."""
    writer_kwargs = {}
    if getattr(args, "sanitize_netcdf_names", False):
        writer_kwargs["sanitize_names"] = True
    return writer_kwargs


def _resolve_protocol_path(args, default_path: str) -> Path:
    """Resolve processing protocol path."""
    if isinstance(args.processing_protocol, str):
        return Path(args.processing_protocol)
    return Path(default_path)


def _format_unit_conversions(conversions):
    """Normalize unit conversions into a per-variable mapping."""
    if not conversions:
        return None
    if isinstance(conversions, dict):
        return conversions
    if isinstance(conversions, list):
        by_var: dict[str, list[str]] = {}
        for item in conversions:
            if isinstance(item, str):
                if ": " in item:
                    var, conv = item.split(": ", 1)
                elif ":" in item:
                    var, conv = item.split(":", 1)
                    conv = conv.strip()
                else:
                    var, conv = "unknown", item
                by_var.setdefault(var, []).append(conv)
            else:
                by_var.setdefault("unknown", []).append(str(item))
        return by_var
    return conversions


def _write_processing_protocol(
    path: Path,
    metadata: dict | None,
    dataset,
    args: argparse.Namespace,
    command: str,
) -> None:
    """Write a processing protocol JSON file for reproducibility."""
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": command,
        "input_file": getattr(args, "input", None),
        "output_file": getattr(args, "output", None),
        "output_format": getattr(args, "output_format", None),
        "pipeline_profile": getattr(args, "pipeline_profile", None),
        "pipeline_file": getattr(args, "pipeline_file", None),
        "pipeline_apply_stages": getattr(args, "pipeline_apply_stages", None),
        "pipeline_skip_stages": getattr(args, "pipeline_skip_stages", None),
        "pipeline_apply_handlers": getattr(args, "pipeline_apply_handlers", None),
        "pipeline_skip_handlers": getattr(args, "pipeline_skip_handlers", None),
        "raw_only": getattr(args, "raw_only", False),
        "reader_args": getattr(args, "reader_args", None),
    }

    if metadata:
        protocol["stages_applied"] = metadata.get("stages_applied")
        protocol["handlers_applied"] = metadata.get("handlers_applied")
        protocol["variable_mappings"] = metadata.get("variable_mappings")
        protocol["derived_parameters"] = metadata.get("derived_parameters")
        protocol["transformations"] = metadata.get("transformations")
        unit_conversions = _format_unit_conversions(metadata.get("unit_conversions"))
        if unit_conversions is not None:
            protocol["unit_conversions"] = unit_conversions
        protocol["unit_validation_issues"] = metadata.get("unit_validation_issues")
        protocol["warnings"] = metadata.get("warnings")

        registry = metadata.get("_metadata_registry")
        if registry is not None and hasattr(registry, "to_dict"):
            protocol["metadata_registry"] = registry.to_dict()
            if hasattr(registry, "sources"):
                protocol["metadata_sources"] = registry.sources()

    try:
        protocol["data_variables"] = list(dataset.data_vars)
        protocol["coordinates"] = list(dataset.coords)
        protocol["dimensions"] = dict(dataset.sizes)
    except Exception:
        logger.debug("Failed to attach dataset structure to processing protocol", exc_info=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)


class ConvertCommand(BaseCommand):
    """Handle file conversion with lazy loading."""

    def execute(self, args: argparse.Namespace) -> CommandResult:
        """Execute convert command."""
        try:
            # Validate and parse parameter mapping if provided
            mapping_dict = None
            if args.mapping:
                import seasenselib.parameters as params
                allowed_parameters = params.allowed_parameters()
                mapping_dict = {}

                for mapping in args.mapping:
                    if '=' not in mapping:
                        raise ValidationError(
                            f"Invalid mapping format: {mapping}. Use 'name=value'")

                    # CLI format is: canonical=original (e.g., temperature=tv290C)
                    # Swap to internal format: original=canonical
                    canonical, original = mapping.split('=', 1)
                    if canonical not in allowed_parameters:
                        raise ValidationError(
                            f"Unknown parameter name: {canonical}. "
                            f"Allowed parameters are: {', '.join(allowed_parameters)}"
                        )
                    # Store as original -> canonical for internal use
                    mapping_dict[original] = canonical

            # Read data with reader-specific kwargs
            reader_kwargs = _build_reader_kwargs(args)
            # Add mapping if provided
            if mapping_dict:
                reader_kwargs['mapping'] = mapping_dict
            # Add stage control kwargs
            reader_kwargs.update(_build_stage_kwargs(args))
            user_metadata = _parse_user_metadata(args)
            if user_metadata:
                reader_kwargs['user_metadata'] = user_metadata

            # Enable metadata capture if processing protocol requested
            want_protocol = getattr(args, 'processing_protocol', None) is not None
            if want_protocol:
                reader_kwargs['return_metadata'] = True

            data = self.io.read_data(args.input, args.input_format, args.header_input, **reader_kwargs)
            metadata = None
            if want_protocol and isinstance(data, tuple):
                data, metadata = data

            if not data:
                raise ValidationError('No data found in file.')

            # Write data
            writer_kwargs = _build_writer_kwargs(args)
            self.io.write_data(data, args.output, args.output_format, **writer_kwargs)

            # Write processing protocol if requested
            if want_protocol:
                protocol_path = _resolve_protocol_path(
                    args,
                    f"{args.output}.processing.protocol.json"
                )
                _write_processing_protocol(protocol_path, metadata, data, args, "convert")

            return CommandResult(success=True, 
                    message=f"Successfully converted {args.input} to {args.output}")

        except Exception as e:
            return CommandResult(success=False, message=str(e))



class ShowCommand(BaseCommand):
    """Handle data inspection with lazy loading."""

    def execute(self, args: argparse.Namespace) -> CommandResult:
        """Execute show command."""
        try:
            # Validate and parse parameter mapping if provided
            mapping_dict = None
            if getattr(args, 'mapping', None):
                import seasenselib.parameters as params
                allowed_parameters = params.allowed_parameters()
                mapping_dict = {}

                for mapping in args.mapping:
                    if '=' not in mapping:
                        raise ValidationError(
                            f"Invalid mapping format: {mapping}. Use 'name=value'")

                    # CLI format is: canonical=original (e.g., temperature=tv290C)
                    # Swap to internal format: original=canonical
                    canonical, original = mapping.split('=', 1)
                    if canonical not in allowed_parameters:
                        raise ValidationError(
                            f"Unknown parameter name: {canonical}. "
                            f"Allowed parameters are: {', '.join(allowed_parameters)}"
                        )
                    # Store as original -> canonical for internal use
                    mapping_dict[original] = canonical

            # Read data with reader-specific kwargs
            reader_kwargs = _build_reader_kwargs(args)
            # Add mapping if provided
            if mapping_dict:
                reader_kwargs['mapping'] = mapping_dict
            # Add layer control kwargs
            reader_kwargs.update(_build_stage_kwargs(args))
            user_metadata = _parse_user_metadata(args)
            if user_metadata:
                reader_kwargs['user_metadata'] = user_metadata

            # Enable metadata capture if processing protocol requested
            want_protocol = getattr(args, 'processing_protocol', None) is not None
            if want_protocol:
                reader_kwargs['return_metadata'] = True
            
            data = self.io.read_data(args.input, args.input_format, args.header_input, **reader_kwargs)
            metadata = None
            if want_protocol and isinstance(data, tuple):
                data, metadata = data

            if not data:
                raise ValidationError('No data found in file.')

            # Display based on schema
            if args.schema == 'summary':
                print(data)
            elif args.schema == 'info':
                data.info()
            elif args.schema == 'example':
                _print_dataset_example(data)

            # Write processing protocol if requested
            if want_protocol:
                protocol_path = _resolve_protocol_path(
                    args,
                    f"{args.input}.processing.protocol.json"
                )
                _write_processing_protocol(protocol_path, metadata, data, args, "show")

            return CommandResult(success=True, message="Data displayed successfully")

        except Exception as e:
            print(f"{e}")
            return CommandResult(success=False, message=str(e))


class SubsetCommand(BaseCommand):
    """Handle data subsetting with lazy loading."""

    def execute(self, args: argparse.Namespace) -> CommandResult:
        """Execute subset command."""
        try:
            # Lazy import processors
            from ...processors import SubsetProcessor

            # Read data with reader-specific kwargs
            reader_kwargs = _build_reader_kwargs(args)
            user_metadata = _parse_user_metadata(args)
            if user_metadata:
                reader_kwargs['user_metadata'] = user_metadata
            data = self.io.read_data(args.input, args.input_format, args.header_input, **reader_kwargs)

            if not data:
                raise ValidationError('No data found in file.')

            # Create subsetter
            subsetter = SubsetProcessor(data)

            # Apply subsetting parameters
            if args.sample_min:
                subsetter.set_sample_min(args.sample_min)
            if args.sample_max:
                subsetter.set_sample_max(args.sample_max)
            if args.time_min:
                subsetter.set_time_min(args.time_min)
            if args.time_max:
                subsetter.set_time_max(args.time_max)
            if args.parameter:
                subsetter.set_parameter_name(args.parameter)
            if args.value_min:
                subsetter.set_parameter_value_min(args.value_min)
            if args.value_max:
                subsetter.set_parameter_value_max(args.value_max)

            # Get subset
            subset = subsetter.get_subset()

            # Output or write
            if args.output:
                writer_kwargs = _build_writer_kwargs(args)
                self.io.write_data(
                    subset,
                    args.output,
                    args.output_format,
                    **writer_kwargs,
                )
                return CommandResult(success=True, message=f"Subset written to {args.output}")
            else:
                print(subset)
                return CommandResult(success=True, message="Subset displayed successfully")

        except Exception as e:
            return CommandResult(success=False, message=str(e))


class CalcCommand(BaseCommand):
    """Handle calculations with lazy loading."""
  
    def execute(self, args: argparse.Namespace) -> CommandResult:
        """Execute calc command."""
        try:
            # Lazy import processors
            from ...processors import ResampleProcessor, StatisticsProcessor
            
            # Read data with reader-specific kwargs
            reader_kwargs = _build_reader_kwargs(args)
            user_metadata = _parse_user_metadata(args)
            if user_metadata:
                reader_kwargs['user_metadata'] = user_metadata
            data = self.io.read_data(args.input, args.input_format, args.header_input, **reader_kwargs)

            if not data:
                raise ValidationError('No data found in file.')

            # Handle resampling if requested
            if args.resample:
                if not args.time_interval:
                    raise ValidationError("Time interval is required when resampling")

                resampler = ResampleProcessor(data)
                data = resampler.resample(args.time_interval)
                
                # Process resampled data
                # pylint: disable=C0415
                import re
                import pandas as pd

                # Format datetime output based on time interval
                datetime_format_pattern = "%Y-%m-%d %H:%M:%S"
                if re.match(r"^[0-9\.]*M$", args.time_interval):
                    datetime_format_pattern = '%Y-%m'
                elif re.match(r"^[0-9\.]*Y$", args.time_interval):
                    datetime_format_pattern = '%Y'
                elif re.match(r"^[0-9\.]*D$", args.time_interval):
                    datetime_format_pattern = '%Y-%m-%d'
                elif re.match(r"^[0-9\.]*H$", args.time_interval):
                    datetime_format_pattern = '%Y-%m-%d %H:%M'
                elif re.match(r"^[0-9\.]*min$", args.time_interval):
                    datetime_format_pattern = '%Y-%m-%d %H:%M'

                # Process each time period
                for time_period, group in data:
                    result = self._run_calculation(
                        group, args.method, args.parameter, StatisticsProcessor)
                    dt_datetime = pd.to_datetime(time_period)
                    datetime_string = dt_datetime.strftime(datetime_format_pattern)
                    print(f"{datetime_string}: {result}")
            else:
                # Single calculation
                result = self._run_calculation(data, args.method, args.parameter, StatisticsProcessor)
                print(result)
            
            return CommandResult(success=True, message="Calculation completed successfully")
            
        except Exception as e:
            return CommandResult(success=False, message=str(e))
    
    def _run_calculation(self, data, method, parameter, StatisticsProcessor):
        """Run the specified calculation on the data."""
        calc = StatisticsProcessor(data, parameter)

        if method == 'max':
            return calc.max()
        elif method == 'min':
            return calc.min()
        elif method == 'mean':
            return calc.mean()
        elif method == 'median':
            return calc.median()
        elif method in ['std', 'standard_deviation']:
            return calc.std()
        elif method in ['var', 'variance']:
            return calc.var()
        else:
            raise ValidationError(f"Unknown calculation method: {method}")
