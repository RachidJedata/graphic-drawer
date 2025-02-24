'use client';
import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Parameters } from './utils';

interface SignalData {
  parameters: Parameters;
  time: number[];
  signal: number[];
  frequency?: number[];
}

interface SignalVisualizationProps {
  endpoint: string;
  params: Parameters;
  title?: string;
}

const formatParameterValue = (value: number): string => {
  const absValue = Math.abs(value);

  if (absValue >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (absValue >= 1) return value.toFixed(3);
  if (value !== 0) {
    const milliValue = value * 1000;
    return milliValue % 1 === 0 ? 
      `${milliValue.toFixed(0)}ms` : 
      `${milliValue.toFixed(1)}ms`;
  }
  return '0';
};

const SignalVisualization = ({ endpoint, params, title }: SignalVisualizationProps) => {
  const d3Container = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<SignalData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const queryParams = new URLSearchParams(
          Object.entries(params).map(([k, v]) => [k, v.toString()])
        );
        const response = await fetch(`http://127.0.0.1:8000/${endpoint}?${queryParams}`);
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const jsonData: SignalData = await response.json();
        setData(jsonData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [endpoint, params]);

  useEffect(() => {
    if (!data || !d3Container.current) return;

    // Clear previous content
    const container = d3.select(d3Container.current);
    container.selectAll('*').remove();

    // Create SVG container
    const svg = container.append('svg')
      .attr('viewBox', '0 0 800 400')
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('background', '#f8f9fa');

    // Create parameter box
    const paramDiv = container.append('div')
      .attr('class', 'parameter-box');

    paramDiv.selectAll('.param-row')
      .data(Object.entries(data.parameters))
      .join('div')
      .attr('class', 'param-row')
      .html(([key, value]) => `
        <span class="param-key">${key}:</span>
        <span class="param-value">${formatParameterValue(value)}</span>
      `);

    // Add styling
    container.append('style')
      .text(`
        .parameter-box {
          position: absolute;
          top: 20px;
          right: 20px;
          background: rgba(255, 255, 255, 0.9);
          padding: 12px;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          z-index: 2;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;
        }
        .param-row {
          display: flex;
          justify-content: space-between;
          gap: 20px;
          margin: 4px 0;
          font-size: 14px;
        }
        .param-key {
          font-weight: 500;
          color: #2d3748;
        }
        .param-value {
          color: #4a5568;
          font-family: 'SF Mono', Menlo, monospace;
        }
      `);

    // Visualization setup
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 800;
    const height = 400;

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([d3.min(data.time)!, d3.max(data.time)!])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([d3.min(data.signal)! - 0.1, d3.max(data.signal)! + 0.1])
      .range([height - margin.bottom, margin.top]);

    // Create line generator
    const isBinary = data.signal.every(v => v === 0 || v === 1);
    const line = d3.line<number>()
      .x((_, i) => xScale(data.time[i]))
      .y(d => yScale(d))
      .curve(isBinary ? d3.curveStep : d3.curveMonotoneX);

    // Draw path
    svg.append('path')
      .datum(data.signal)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 1.5)
      .attr('d', line);

    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.format('.3s'));
    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(v => isBinary ? `${v}` : d3.format('.2f')(v as number));

    svg.append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(xAxis);

    svg.append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(yAxis);

    // Axis labels
    svg.append('text')
      .attr('transform', `translate(${width / 2}, ${height - 5})`)
      .style('text-anchor', 'middle')
      .text('Time (s)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 15)
      .attr('x', -height / 2)
      .style('text-anchor', 'middle')
      .text('Amplitude');

  }, [data]);

  if (loading) return <div className="p-4 text-gray-600">Loading...</div>;
  if (error) return <div className="p-4 text-red-600">Error: {error}</div>;

  return (
    <div className="w-full max-w-4xl mx-auto relative">
      {title && <h2 className="text-xl font-semibold mb-4 text-gray-800">{title}</h2>}
      <div ref={d3Container} className="w-full h-[400px] relative" />
    </div>
  );
};

export default SignalVisualization;