import { Component, input, output } from '@angular/core';
import { Feature } from '../../_models/feature';

@Component({
  selector: 'app-accordion-item',
  standalone: true,
  imports: [],
  templateUrl: './accordion-item.component.html',
  styleUrl: './accordion-item.component.css',
})
export class AccordionItemComponent {
  feature = input.required<Feature>();
  accordionId = input.required<string>();
  opened = output<void>();

  toggleOpen() {
    this.opened.emit();
  }
}
