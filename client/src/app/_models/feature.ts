import { Rule } from './rule';

export interface Feature {
  title: string;
  description: string;
  rules: Rule[];
  example: string;
}
