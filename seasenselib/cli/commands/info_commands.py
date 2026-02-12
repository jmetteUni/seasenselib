"""
Information commands (list, formats).
"""

import argparse
import csv
import json
from io import StringIO
from .base import BaseCommand, CommandResult


class ListCommand(BaseCommand):
    """Handle listing of readers, writers, and plotters with minimal dependencies."""

    def execute(self, args: argparse.Namespace) -> CommandResult:
        """Execute list command."""
        try:
            resource_type = args.resource_type
            all_data = []

            if resource_type == 'parameters':
                all_data = self._list_parameters()
                self._apply_filters_and_sort(all_data, args, resource_type)
                self._output_parameters(all_data, args)
                return CommandResult(success=True, message="Parameter list displayed successfully")

            if resource_type == 'pipeline-stages':
                all_data = self._list_stages()
                self._apply_filters_and_sort(all_data, args, resource_type)
                self._output_stages(all_data, args)
                return CommandResult(success=True, message="Stage list displayed successfully")

            if resource_type == 'pipeline-handlers':
                all_data = self._list_handlers()
                self._apply_filters_and_sort(all_data, args, resource_type)
                self._output_handlers(all_data, args)
                return CommandResult(success=True, message="Handler list displayed successfully")

            if resource_type == 'pipeline-profiles':
                all_data = self._list_pipeline_profiles()
                self._apply_filters_and_sort(all_data, args, resource_type)
                self._output_pipeline_profiles(all_data, args)
                return CommandResult(success=True, message="Pipeline profile list displayed successfully")

            # Use autodiscovery to get format information
            # pylint: disable=C0415
            from ...core.autodiscovery import ReaderDiscovery, WriterDiscovery, PlotterDiscovery

            # Collect data based on resource type
            if resource_type in ['all', 'readers']:
                discovery = ReaderDiscovery()
                readers = self._convert_format_info(discovery.get_format_info(), 'reader')
                all_data.extend(readers)

            if resource_type in ['all', 'writers']:
                discovery = WriterDiscovery()
                writers = self._convert_format_info(discovery.get_format_info(), 'writer')
                all_data.extend(writers)

            if resource_type in ['all', 'plotters']:
                discovery = PlotterDiscovery()
                plotters = self._convert_format_info(discovery.get_format_info(), 'plotter')
                all_data.extend(plotters)

            # Apply filtering and sorting
            self._apply_filters_and_sort(all_data, args, resource_type)

            # Output based on selected format
            self._output_list(all_data, args)

            return CommandResult(success=True, message="List displayed successfully")

        except Exception as e:
            return CommandResult(success=False, message=str(e))

    def _convert_format_info(self, format_info_list, resource_type):
        """Convert format info to unified structure with type."""
        result = []
        for format_info in format_info_list:
            item = {
                'name': format_info.get('name', 'Unknown'),
                'key': format_info['key'],
                'extension': format_info.get('extension') or '',
                'class': format_info['class_name'],
                'type': resource_type,
                'is_plugin': format_info.get('is_plugin', False)
            }
            result.append(item)
        return result

    def _list_parameters(self):
        """List canonical parameter names."""
        try:
            # pylint: disable=C0415
            import seasenselib.parameters as params
            metadata = getattr(params, 'metadata', {}) or {}
            default_mappings = getattr(params, 'default_mappings', {}) or {}
            allowed = params.allowed_parameters()
        except Exception:
            metadata = {}
            default_mappings = {}
            allowed = {}

        data = []
        names = set()
        if isinstance(default_mappings, dict):
            names.update(default_mappings.keys())
        if isinstance(metadata, dict):
            names.update(metadata.keys())

        if names:
            for name in sorted(names):
                if isinstance(metadata, dict) and name in metadata:
                    description = self._describe_parameter(name, metadata[name])
                else:
                    description = allowed.get(name) if isinstance(allowed, dict) else None
                    if not description:
                        description = name.replace('_', ' ').title()
                data.append({'name': name, 'description': description})
        else:
            for name, description in allowed.items():
                data.append({
                    'name': name,
                    'description': description or ''
                })
        return data

    @staticmethod
    def _describe_parameter(name, info):
        """Build a short description from metadata."""
        if not isinstance(info, dict):
            return name.replace('_', ' ').title()
        long_name = info.get('long_name') or name.replace('_', ' ').title()
        units = info.get('units')
        if units:
            return f"{long_name} ({units})"
        return long_name

    def _apply_filters_and_sort(self, data, args, resource_type):
        """Apply filtering and sorting to list data."""
        if args.filter:
            filter_term = args.filter.lower()
            if resource_type == 'parameters':
                data[:] = [
                    item for item in data
                    if filter_term in item['name'].lower() or
                       filter_term in item.get('description', '').lower()
                ]
            elif resource_type == 'pipeline-stages':
                data[:] = [
                    item for item in data
                    if filter_term in item.get('name', '').lower() or
                       filter_term in item.get('class', '').lower()
                ]
            elif resource_type == 'pipeline-handlers':
                data[:] = [
                    item for item in data
                    if filter_term in item.get('name', '').lower() or
                       filter_term in item.get('stage', '').lower() or
                       filter_term in item.get('class', '').lower()
                ]
            elif resource_type == 'pipeline-profiles':
                data[:] = [
                    item for item in data
                    if filter_term in item.get('name', '').lower() or
                       filter_term in item.get('description', '').lower() or
                       filter_term in item.get('file', '').lower()
                ]
            else:
                data[:] = [
                    item for item in data
                    if filter_term in item['name'].lower() or
                       filter_term in item.get('extension', '').lower() or
                       filter_term in item['key'].lower() or
                       filter_term in item['type'].lower()
                ]

        sort_key = args.sort
        if resource_type == 'parameters':
            if sort_key in ['name', 'key']:
                data.sort(key=lambda x: x['name'].lower(), reverse=args.reverse)
        elif resource_type == 'pipeline-stages':
            if sort_key == 'name':
                data.sort(key=lambda x: x.get('name', '').lower(), reverse=args.reverse)
            elif sort_key == 'class':
                data.sort(key=lambda x: x.get('class', '').lower(), reverse=args.reverse)
        elif resource_type == 'pipeline-handlers':
            if sort_key == 'stage':
                data.sort(key=lambda x: x.get('stage', '').lower(), reverse=args.reverse)
            elif sort_key == 'name':
                data.sort(key=lambda x: x.get('name', '').lower(), reverse=args.reverse)
            elif sort_key == 'class':
                data.sort(key=lambda x: x.get('class', '').lower(), reverse=args.reverse)
        elif resource_type == 'pipeline-profiles':
            if sort_key == 'name':
                data.sort(key=lambda x: x.get('name', '').lower(), reverse=args.reverse)
        else:
            if sort_key == 'name':
                data.sort(key=lambda x: x['name'].lower(), reverse=args.reverse)
            elif sort_key == 'key':
                data.sort(key=lambda x: x['key'].lower(), reverse=args.reverse)
            elif sort_key == 'extension':
                data.sort(key=lambda x: x.get('extension', '').lower(), reverse=args.reverse)
            elif sort_key == 'type':
                data.sort(key=lambda x: x['type'].lower(), reverse=args.reverse)

    def _list_stages(self):
        """List available pipeline stages."""
        try:
            from ...pipeline.registry import StageRegistry
            registry = StageRegistry.get_instance()
            builtin = set(registry.list_builtin_stages())
            data = []
            for name in registry.list_stages():
                cls = registry.get_stage_class(name)
                data.append({
                    'name': name,
                    'class': cls.__name__,
                    'class_path': f"{cls.__module__}.{cls.__name__}",
                    'is_plugin': name not in builtin,
                })
            return data
        except Exception:
            return []

    def _list_handlers(self):
        """List available pipeline handlers."""
        try:
            from ...pipeline.handler_catalog import list_handlers
            return list_handlers(include_plugins=True)
        except Exception:
            return []

    def _list_pipeline_profiles(self):
        """List available pipeline profiles."""
        try:
            from importlib import resources
            profiles = []
            root = resources.files('seasenselib.config.pipeline')
            for item in root.iterdir():
                if not item.is_file():
                    continue
                suffix = item.suffix.lower()
                if suffix not in {'.json', '.yaml', '.yml', '.toml'}:
                    continue
                description = ""
                if suffix == '.json':
                    try:
                        with item.open('r', encoding='utf-8') as handle:
                            data = json.load(handle)
                        if isinstance(data, dict):
                            global_cfg = data.get('global', {})
                            if isinstance(global_cfg, dict):
                                description = str(global_cfg.get('description', '') or '')
                            if not description:
                                description = str(data.get('description', '') or '')
                    except Exception:
                        description = ""
                profiles.append({'name': item.stem, 'description': description, 'file': item.name})
            return profiles
        except Exception:
            return []

    def _output_list(self, data, args):
        """Output list in the requested format."""
        output_format = args.output

        if output_format == 'json':
            print(json.dumps(data, indent=2))
        elif output_format == 'yaml':
            try:
                # pylint: disable=C0415
                import yaml
                print(yaml.dump(data, default_flow_style=False))
            except ImportError:
                print("Error: PyYAML not installed. Install with: pip install PyYAML")
                print("Falling back to JSON format:")
                print(json.dumps(data, indent=2))
        elif output_format == 'csv':
            self._output_csv(data, args)
        else:  # table format (default)
            self._output_table(data, args)

    def _output_parameters(self, data, args):
        """Output parameter list in the requested format."""
        output_format = args.output

        if output_format == 'json':
            print(json.dumps(data, indent=2))
        elif output_format == 'yaml':
            try:
                # pylint: disable=C0415
                import yaml
                print(yaml.dump(data, default_flow_style=False))
            except ImportError:
                print("Error: PyYAML not installed. Install with: pip install PyYAML")
                print("Falling back to JSON format:")
                print(json.dumps(data, indent=2))
        elif output_format == 'csv':
            self._output_parameters_csv(data, args)
        else:
            self._output_parameters_table(data, args)

    def _output_stages(self, data, args):
        """Output stage list in the requested format."""
        output_format = args.output

        if output_format == 'json':
            print(json.dumps(data, indent=2))
        elif output_format == 'yaml':
            try:
                # pylint: disable=C0415
                import yaml
                print(yaml.dump(data, default_flow_style=False))
            except ImportError:
                print("Error: PyYAML not installed. Install with: pip install PyYAML")
                print("Falling back to JSON format:")
                print(json.dumps(data, indent=2))
        elif output_format == 'csv':
            self._output_stages_csv(data, args)
        else:
            self._output_stages_table(data, args)

    def _output_handlers(self, data, args):
        """Output handler list in the requested format."""
        output_format = args.output

        if output_format == 'json':
            print(json.dumps(data, indent=2))
        elif output_format == 'yaml':
            try:
                # pylint: disable=C0415
                import yaml
                print(yaml.dump(data, default_flow_style=False))
            except ImportError:
                print("Error: PyYAML not installed. Install with: pip install PyYAML")
                print("Falling back to JSON format:")
                print(json.dumps(data, indent=2))
        elif output_format == 'csv':
            self._output_handlers_csv(data, args)
        else:
            self._output_handlers_table(data, args)

    def _output_pipeline_profiles(self, data, args):
        """Output pipeline profile list in the requested format."""
        output_format = args.output

        if output_format == 'json':
            print(json.dumps(data, indent=2))
        elif output_format == 'yaml':
            try:
                # pylint: disable=C0415
                import yaml
                print(yaml.dump(data, default_flow_style=False))
            except ImportError:
                print("Error: PyYAML not installed. Install with: pip install PyYAML")
                print("Falling back to JSON format:")
                print(json.dumps(data, indent=2))
        elif output_format == 'csv':
            self._output_pipeline_profiles_csv(data, args)
        else:
            self._output_pipeline_profiles_table(data, args)

    def _output_parameters_csv(self, data, args):
        """Output parameter data as CSV."""
        output = StringIO()
        fieldnames = ['name', 'description']
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        if not args.no_header:
            writer.writeheader()
        for item in data:
            row = {k: item.get(k, '') for k in fieldnames}
            writer.writerow(row)
        print(output.getvalue().rstrip())

    def _output_stages_csv(self, data, args):
        """Output stage data as CSV."""
        output = StringIO()
        fieldnames = ['name', 'class', 'is_plugin']
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        if not args.no_header:
            writer.writeheader()
        for item in data:
            row = {k: item.get(k, '') for k in fieldnames}
            writer.writerow(row)
        print(output.getvalue().rstrip())

    def _output_handlers_csv(self, data, args):
        """Output handler data as CSV."""
        output = StringIO()
        fieldnames = ['stage', 'name', 'class', 'is_plugin']
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        if not args.no_header:
            writer.writeheader()
        for item in data:
            row = {k: item.get(k, '') for k in fieldnames}
            writer.writerow(row)
        print(output.getvalue().rstrip())

    def _output_pipeline_profiles_csv(self, data, args):
        """Output pipeline profile data as CSV."""
        output = StringIO()
        fieldnames = ['name', 'description', 'file']
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        if not args.no_header:
            writer.writeheader()
        for item in data:
            row = {k: item.get(k, '') for k in fieldnames}
            writer.writerow(row)
        print(output.getvalue().rstrip())

    def _output_parameters_table(self, data, args):
        """Output parameter data as a formatted table."""
        if not data:
            print("No parameters found matching the criteria.")
            return

        columns = [
            ('Name', 'name'),
            ('Description', 'description'),
        ]

        col_widths = []
        for header, field in columns:
            max_width = len(header)
            for item in data:
                value = str(item.get(field, ''))
                max_width = max(max_width, len(value))
            col_widths.append(max_width + 2)

        border = "+" + "+".join("-" * width for width in col_widths) + "+"

        if not args.no_header:
            print(border)
            header_row = "|"
            for i, (header, _) in enumerate(columns):
                header_row += f" {header:<{col_widths[i]-2}} |"
            print(header_row)
            print(border)

        for item in data:
            row = "|"
            for i, (_, field) in enumerate(columns):
                value = str(item.get(field, ''))
                row += f" {value:<{col_widths[i]-2}} |"
            print(row)

        if not args.no_header:
            print(border)
            print(f"\nTotal: {len(data)} parameter(s)")

    def _output_stages_table(self, data, args):
        """Output stage data as a formatted table."""
        if not data:
            print("No stages found matching the criteria.")
            return

        columns = [('Name', 'name'), ('Plugin', 'is_plugin')]
        if getattr(args, 'list_details', False):
            columns.insert(1, ('Class', 'class'))

        col_widths = []
        for header, field in columns:
            max_width = len(header)
            for item in data:
                value = str(item.get(field, ''))
                if field == 'is_plugin':
                    value = 'Yes' if item.get('is_plugin', False) else 'No'
                max_width = max(max_width, len(value))
            col_widths.append(max_width + 2)

        border = "+" + "+".join("-" * width for width in col_widths) + "+"

        if not args.no_header:
            print(border)
            header_row = "|"
            for i, (header, _) in enumerate(columns):
                header_row += f" {header:<{col_widths[i]-2}} |"
            print(header_row)
            print(border)

        for item in data:
            row = "|"
            for i, (_, field) in enumerate(columns):
                value = str(item.get(field, ''))
                if field == 'is_plugin':
                    value = 'Yes' if item.get('is_plugin', False) else 'No'
                row += f" {value:<{col_widths[i]-2}} |"
            print(row)

        if not args.no_header:
            print(border)
            print(f"\nTotal: {len(data)} stage(s)")

    def _output_handlers_table(self, data, args):
        """Output handler data as a formatted table."""
        if not data:
            print("No handlers found matching the criteria.")
            return

        columns = [('Stage', 'stage'), ('Handler', 'name'), ('Plugin', 'is_plugin')]
        if getattr(args, 'list_details', False):
            columns.insert(2, ('Class', 'class'))

        col_widths = []
        for header, field in columns:
            max_width = len(header)
            for item in data:
                value = str(item.get(field, ''))
                if field == 'is_plugin':
                    value = 'Yes' if item.get('is_plugin', False) else 'No'
                max_width = max(max_width, len(value))
            col_widths.append(max_width + 2)

        border = "+" + "+".join("-" * width for width in col_widths) + "+"

        if not args.no_header:
            print(border)
            header_row = "|"
            for i, (header, _) in enumerate(columns):
                header_row += f" {header:<{col_widths[i]-2}} |"
            print(header_row)
            print(border)

        for item in data:
            row = "|"
            for i, (_, field) in enumerate(columns):
                value = str(item.get(field, ''))
                if field == 'is_plugin':
                    value = 'Yes' if item.get('is_plugin', False) else 'No'
                row += f" {value:<{col_widths[i]-2}} |"
            print(row)

        if not args.no_header:
            print(border)
            print(f"\nTotal: {len(data)} handler(s)")

    def _output_pipeline_profiles_table(self, data, args):
        """Output pipeline profile data as a formatted table."""
        if not data:
            print("No pipeline profiles found matching the criteria.")
            return

        columns = [('Name', 'name'), ('Description', 'description'), ('File', 'file')]

        col_widths = []
        for header, field in columns:
            max_width = len(header)
            for item in data:
                value = str(item.get(field, ''))
                max_width = max(max_width, len(value))
            col_widths.append(max_width + 2)

        border = "+" + "+".join("-" * width for width in col_widths) + "+"

        if not args.no_header:
            print(border)
            header_row = "|"
            for i, (header, _) in enumerate(columns):
                header_row += f" {header:<{col_widths[i]-2}} |"
            print(header_row)
            print(border)

        for item in data:
            row = "|"
            for i, (_, field) in enumerate(columns):
                value = str(item.get(field, ''))
                row += f" {value:<{col_widths[i]-2}} |"
            print(row)

        if not args.no_header:
            print(border)
            print(f"\nTotal: {len(data)} profile(s)")
    def _output_csv(self, data, args):
        """Output data as CSV."""
        output = StringIO()
        fieldnames = ['name', 'key', 'type', 'extension']
        if getattr(args, 'list_details', False):
            fieldnames.extend(['class', 'is_plugin'])

        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        if not args.no_header:
            writer.writeheader()

        for item in data:
            row = {k: item.get(k, '') for k in fieldnames}
            writer.writerow(row)

        print(output.getvalue().rstrip())

    def _output_table(self, data, args):
        """Output data as a formatted table."""
        if not data:
            print("No resources found matching the criteria.")
            return

        # Determine if we're showing a single resource type
        resource_type = args.resource_type
        single_type = resource_type in ['readers', 'writers', 'plotters']
        
        # Determine columns to show
        columns = [
            ('Name', 'name'),
            ('Key', 'key'),
        ]
        
        # Only show Type column if showing multiple types (all)
        if not single_type:
            columns.append(('Type', 'type'))
        
        # Only show Extension column for readers/writers (not for plotters)
        if resource_type != 'plotters':
            # Check if any item has an extension
            has_extensions = any(item.get('extension') for item in data)
            if has_extensions:
                columns.append(('Extension', 'extension'))
        
        # Add Plugin column
        columns.append(('Plugin', 'is_plugin'))
        
        if getattr(args, 'list_details', False):
            columns.append(('Class', 'class'))

        # Calculate column widths
        col_widths = []
        for header, field in columns:
            max_width = len(header)
            for item in data:
                value = str(item.get(field, ''))
                # Format plugin column as Yes/No
                if field == 'is_plugin':
                    value = 'Yes' if item.get('is_plugin', False) else 'No'
                max_width = max(max_width, len(value))
            col_widths.append(max_width + 2)  # Add padding

        # Create table border
        border = "+" + "+".join("-" * width for width in col_widths) + "+"

        # Print table
        if not args.no_header:
            print(border)
            header_row = "|"
            for i, (header, _) in enumerate(columns):
                header_row += f" {header:<{col_widths[i]-2}} |"
            print(header_row)
            print(border)

        for item in data:
            row = "|"
            for i, (_, field) in enumerate(columns):
                value = str(item.get(field, ''))
                # Format plugin column as Yes/No
                if field == 'is_plugin':
                    value = 'Yes' if item.get('is_plugin', False) else 'No'
                row += f" {value:<{col_widths[i]-2}} |"
            print(row)

        if not args.no_header:
            print(border)

        # Show summary
        if not args.no_header:
            total = len(data)
            plugins = sum(1 for item in data if item.get('is_plugin', False))
            print(f"\nTotal: {total} resource(s)", end='')
            if plugins > 0:
                print(f" ({plugins} plugin(s))")
            else:
                print()
            if args.filter:
                print(f"Filtered by: '{args.filter}'")
            
            # Show usage hint if showing all resources (no specific type selected)
            if args.resource_type == 'all' and not args.filter:
                print("\nTip: Use 'seasenselib list readers', 'list writers', or 'list plotters'")
                print("     to show only specific resource types.")
                print("     Use 'seasenselib list parameters' to list canonical variable names.")
                print("     Use 'seasenselib list pipeline-stages' or 'list pipeline-handlers' for pipeline components.")
                print("     Use 'seasenselib list pipeline-profiles' to list built-in pipeline profiles.")
                print("     Use --help for more options (filtering, sorting, output formats).")


class FormatsCommand(BaseCommand):
    """
    Legacy formats command - redirects to ListCommand with 'readers' resource type.
    Maintained for backward compatibility.
    """

    def execute(self, args: argparse.Namespace) -> CommandResult:
        """
        Execute formats command by delegating to ListCommand.
        
        This maintains backward compatibility by treating 'formats' as 
        an alias for 'list readers'.
        """
        # Create a modified args namespace that forces resource_type to 'readers'
        # This ensures 'formats' only shows readers (original behavior)
        list_args = argparse.Namespace(**vars(args))
        list_args.resource_type = 'readers'
        
        # Delegate to ListCommand
        list_command = ListCommand(self.io)
        return list_command.execute(list_args)
