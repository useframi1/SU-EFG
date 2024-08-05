import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Prompt } from '../../_models/prompt';

@Component({
  selector: 'app-prompt-form',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './prompt-form.component.html',
  styleUrl: './prompt-form.component.css',
})
export class PromptFormComponent {
  prompt = input.required<Prompt>();
  awaitingResponse = input.required<boolean>();
  formSubmit = output<void>();

  send_prompt() {
    this.formSubmit.emit();
  }
}
