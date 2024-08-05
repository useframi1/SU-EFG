import { Component, ElementRef, inject, ViewChild } from '@angular/core';
import { AccordionItemComponent } from '../_components/accordion-item/accordion-item.component';
import { Feature } from '../_models/feature';

@Component({
  selector: 'app-instructions',
  standalone: true,
  imports: [AccordionItemComponent],
  templateUrl: './instructions.component.html',
  styleUrl: './instructions.component.css',
})
export class InstructionsComponent {
  @ViewChild('featuresAccordion')
  featuresAccordion = inject(ElementRef);

  features: Feature[] = [
    {
      title: 'Age',
      description: 'Age of the client in years.',
      rules: [
        { rule: 'Must be a positive integer', values: [] },
        { rule: 'Default value is 40', values: [] },
      ],
      example: '53 years old',
    },
    {
      title: 'Gender',
      description: 'The gender of the client.',
      rules: [
        {
          rule: 'Can write He/She, Male/Female, Man/Woman. Anything to highlight the gender',
          values: [],
        },
        {
          rule: 'Default value is Male',
          values: [],
        },
      ],
      example: 'The client is a female.',
    },
    {
      title: 'Risk Rate',
      description: 'The assigned risk rate of the client from the KYC system.',
      rules: [
        {
          rule: 'Must be one of the given values:',
          values: ['Low', 'Medium', 'High', 'Not Assigned'],
        },
        {
          rule: 'Default value is Not Assigned',
          values: [],
        },
      ],
      example: 'The client has a low risk rate.',
    },
    {
      title: 'Average Price',
      description: "The average price of the client's orders.",
      rules: [
        {
          rule: 'Must be a real number',
          values: [],
        },
        {
          rule: 'Default value is 9.56',
          values: [],
        },
      ],
      example: 'The average price for the clients orders is 7.5',
    },
    {
      title: 'Average Order Rate Difference',
      description: 'This is the trend of the clients order rate.',
      rules: [
        {
          rule: 'Must depict one of the given categories:',
          values: ['Decreased', 'Constant', 'Increased'],
        },
        {
          rule: 'Default value is Constant',
          values: [],
        },
      ],
      example: 'The client has done more orders recently.',
    },
    {
      title: 'Average Quantity Ordered Rate Difference',
      description: "This is the trend for the client's order quantity.",
      rules: [
        {
          rule: 'Must depict one of the given categories:',
          values: ['Decreased', 'Constant', 'Increased'],
        },
        {
          rule: 'Default value is Constant',
          values: [],
        },
      ],
      example: 'The client has a decreased order quantity rate.',
    },
    {
      title: 'Completed Orders Ratio',
      description: 'The categorized ratio of the clients completed orders.',
      rules: [
        {
          rule: 'Must depict one of the given categories:',
          values: ['All', 'More Than Half', 'Less Than Half', 'None'],
        },
        {
          rule: 'Default value is More Than Half',
          values: [],
        },
      ],
      example: "More than half of the client's orders are completed.",
    },
    {
      title: 'Canceled Orders Ratio',
      description: 'The categorized ratio of the clients canceled orders.',
      rules: [
        {
          rule: 'Must depict one of the given categories:',
          values: ['All', 'Most', 'Moderate', 'Little', 'None'],
        },
        {
          rule: 'Default value is Moderate',
          values: [],
        },
      ],
      example: "Little of the client's orders are canceled.",
    },
    {
      title: 'Most Frequent Order Type',
      description: 'The most frequent order type of the client.',
      rules: [
        {
          rule: 'Must be one of the given values:',
          values: ['Buy', 'Sell'],
        },
        {
          rule: 'Default value is Sell',
          values: [],
        },
      ],
      example: 'The client most frequently buys.',
    },
    {
      title: 'Most Frequent Execution Status',
      description: "The most frequent execution status of the client's orders.",
      rules: [
        {
          rule: 'Must be one of the given values:',
          values: ['Executed', 'Partially Executed', 'Not Executed'],
        },
        {
          rule: 'Default value is Executed',
          values: [],
        },
      ],
      example: "The client's orders are usually fully executed",
    },
    {
      title: 'Most Frequent Sector Name',
      description: 'The most frequent sector that the client orders in.',
      rules: [
        {
          rule: 'Must be one of the given values:',
          values: [
            'Industries',
            'Financials',
            'Real Estate',
            'Materials',
            'Energy',
            'INVESTMENT',
            'Consumer Discretionary',
            'INDUSTRIAL',
            'Information Technology',
            'Health Care',
            'Consumer Staples',
            'REAL ESTATE',
            'Telecommunication Services',
            'Basic Materials',
            'Others',
            'FOOD',
            'Tourism',
            'Telecommunications',
            'SERVICES',
          ],
        },
        {
          rule: 'Default value is Financials',
          values: [],
        },
      ],
      example: 'The customer mostly orders from the financials sector.',
    },
  ];

  opened(featureTitle: string) {
    setTimeout(() => this.scrollIntoView(featureTitle), 0);
  }

  scrollIntoView(featureTitle: string): void {
    try {
      const accordionElement = this.featuresAccordion.nativeElement;
      const bodyElement = accordionElement.querySelector(
        `#collapse${featureTitle.replaceAll(' ', '')}`
      );

      if (bodyElement && !this.isElementInViewport(bodyElement)) {
        bodyElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    } catch (err) {
      console.log('Scroll to body failed:', err);
    }
  }

  isElementInViewport(el: HTMLElement): boolean {
    const rect = el.getBoundingClientRect();
    const viewHeight = Math.max(
      document.documentElement.clientHeight,
      window.innerHeight
    );
    return rect.top >= 0 && rect.bottom <= viewHeight;
  }
}
