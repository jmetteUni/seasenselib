"""
Plotting commands (plot, plot-ts, plot-profile, plot-series).
"""

import argparse
from ...core.exceptions import ValidationError
from .base import BaseCommand, CommandResult
from .data_commands import _build_reader_kwargs


class PlotCommand(BaseCommand):
    """Handle unified plot command with plotter discovery."""

    def execute(self, args: argparse.Namespace) -> CommandResult:
        """Execute plot command."""
        try:
            # pylint: disable=C0415
            from ...core.autodiscovery import PlotterDiscovery
            
            discovery = PlotterDiscovery()
            
            # If --list-plotters flag is set, list available plotters and exit
            if args.list_plotters:
                return self._list_plotters(discovery)
            
            # Validate that plotter key is provided
            if not args.plotter:
                return CommandResult(
                    success=False, 
                    message="Error: plotter key is required.\n" +
                            "Use 'seasenselib list plotters' to see available plotters.\n" +
                            "Example: seasenselib plot ts-diagram -i data.cnv"
                )
            
            # Find the plotter class
            plotter_class = discovery.get_class_by_key(args.plotter)
            if not plotter_class:
                available = discovery.get_format_info()
                keys = [p['key'] for p in available]
                return CommandResult(
                    success=False,
                    message=f"Error: Unknown plotter '{args.plotter}'.\n" +
                            f"Available plotters: {', '.join(keys)}\n" +
                            "Use 'seasenselib list plotters' for more details."
                )
            
            # Read data with reader-specific kwargs
            reader_kwargs = _build_reader_kwargs(args)
            data = self.io.read_data(
                args.input, 
                args.input_format, 
                args.header_input,
                **reader_kwargs
            )
            
            if not data:
                raise ValidationError('No data found in file.')
            
            # Create plotter instance
            plotter = plotter_class(data)
            
            # Prepare plot arguments based on available args
            plot_kwargs = self._prepare_plot_kwargs(args)
            
            # Call the plot method
            plotter.plot(**plot_kwargs)
            
            message = f"Plot created successfully using {plotter_class.__name__}"
            if args.output:
                message += f" and saved to {args.output}"
            
            return CommandResult(success=True, message=message)
        
        except Exception as e:
            return CommandResult(success=False, message=f"Error: {str(e)}")
    
    def _prepare_plot_kwargs(self, args: argparse.Namespace) -> dict:
        """Prepare plot kwargs from parsed arguments.
        
        This method generically converts all parsed arguments to kwargs that can be
        passed to the plotter. Plotters define their own arguments via add_cli_arguments(),
        so we just need to pass everything along.
        """
        plot_kwargs = {}
        
        # Convert argparse.Namespace to dict and filter out None values and internal args
        args_dict = vars(args)
        
        # Skip internal/common arguments that aren't for the plotter
        skip_args = {
            'plotter', 'input', 'input_format', 'header_input', 'command',
            'list_plotters', 'no_sanitize', 'no_fix_coords', 'reader_args'
        }
        
        for key, value in args_dict.items():
            if key in skip_args or value is None:
                continue
            
            # Map CLI argument names to plotter method parameter names
            # Common mappings that apply to all plotters
            if key == 'output':
                plot_kwargs['output_file'] = value
            elif key == 'parameter':
                # Handle parameter argument - always convert to list for consistency
                if isinstance(value, list):
                    plot_kwargs['parameter_names'] = value
                else:
                    plot_kwargs['parameter_names'] = [value]
            # Handle boolean flags that negate a plotter option
            elif key.startswith('no_') and value is True:
                # Convert no_grid -> show_grid=False, no_isolines -> show_density_isolines=False
                positive_key = key[3:]  # Remove 'no_' prefix
                # Map to the positive form used by plotters
                positive_param = f'show_{positive_key}' if not positive_key.startswith('show_') else positive_key
                if positive_key == 'isolines':
                    positive_param = 'show_density_isolines'
                elif positive_key == 'colormap':
                    positive_param = 'use_colormap'
                plot_kwargs[positive_param] = False
            # Pass through all other arguments as-is
            else:
                plot_kwargs[key] = value
        
        return plot_kwargs
    
    def _list_plotters(self, discovery) -> CommandResult:
        """List available plotters."""
        plotters = discovery.get_format_info()
        
        if not plotters:
            return CommandResult(success=True, message="No plotters available.")
        
        print("\nAvailable plotters:")
        print("-" * 60)
        
        for plotter in sorted(plotters, key=lambda x: x['key']):
            plugin_marker = " [PLUGIN]" if plotter.get('is_plugin', False) else ""
            print(f"  {plotter['key']:<20} {plotter['name']}{plugin_marker}")
        
        print("-" * 60)
        print(f"\nTotal: {len(plotters)} plotter(s)")
        print("\nUsage: seasenselib plot <plotter-key> -i <input-file> [options]")
        print("Example: seasenselib plot ts-diagram -i data.cnv -o output.png")
        
        return CommandResult(success=True, message="")
